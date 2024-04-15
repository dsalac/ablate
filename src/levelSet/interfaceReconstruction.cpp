#include "interfaceReconstruction.hpp"
#include <petsc.h>
#include "utilities/petscSupport.hpp"
#include "utilities/mathUtilities.hpp"
#include "domain/fieldAccessor.hpp"
#include "levelSetUtilities.hpp"



using namespace ablate::levelSet;

#define xexit(S, ...) {PetscFPrintf(MPI_COMM_WORLD, stderr, \
  "\x1b[1m(%s:%d, %s)\x1b[0m\n  \x1b[1m\x1b[90mexiting:\x1b[0m " S "\n", \
  __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__); exit(0);}

static PetscInt FindCell(DM dm, const PetscInt dim, const PetscReal x0[], const PetscInt nCells, const PetscInt cells[], PetscReal *distOut) {
  // Return the cell with the cell-center that is the closest to a given point

  PetscReal dist = PETSC_MAX_REAL;
  PetscInt closestCell = -1;

  for (PetscInt c = 0; c < nCells; ++c) {
    PetscReal x[dim];
    DMPlexComputeCellGeometryFVM(dm, cells[c], NULL, x, NULL) >> ablate::utilities::PetscUtilities::checkError;

    ablate::utilities::MathUtilities::Subtract(dim, x, x0, x);
    PetscReal cellDist = ablate::utilities::MathUtilities::MagVector(dim, x);
    if (cellDist < dist) {
      closestCell = cells[c];
      dist = cellDist;
    }

  }

  if (distOut) *distOut = dist;

  return (closestCell);


}

void Reconstruction::BuildInterpCellList() {

  PetscReal h;
  DMPlexGetMinRadius(cellDM, &h) >> ablate::utilities::PetscUtilities::checkError;
  h *= 2.0; // Min radius returns the distance between a cell-center and a face. Double it to get the average cell size
  const PetscReal sigma = sigmaFactor*h;

  PetscInt dim;
  DMGetDimension(cellDM, &dim) >> ablate::utilities::PetscUtilities::checkError;

  const PetscInt nStencil = PetscPowInt(nQuad, dim); // The number of cells in the integration stencil

  const PetscInt nRange[3] = {nQuad, (dim > 1) ? nQuad : 1, (dim > 2) ? nQuad : 1};

  PetscMalloc1(nStencil*nTotalCell, &interpCellList) >> ablate::utilities::PetscUtilities::checkError;

  for (PetscInt c = 0; c < nTotalCell; ++c) {

    const PetscInt cell = cellList[c];

    PetscReal x0[3] = {0.0, 0.0, 0.0};
    DMPlexComputeCellGeometryFVM(cellDM, cell, NULL, x0, NULL) >> ablate::utilities::PetscUtilities::checkError;

    PetscInt nCells, *cellList;
    DMPlexGetNeighbors(cellDM, cell, 2, -1.0, -1, PETSC_FALSE, PETSC_FALSE, &nCells, &cellList) >> ablate::utilities::PetscUtilities::checkError;

    for (PetscInt i = 0; i < nRange[0]; ++i) {
      for (PetscInt j = 0; j < nRange[1]; ++j) {
        for (PetscInt k = 0; k < nRange[2]; ++k) {

          PetscReal x[3] = {x0[0] + sigma*quad[i], x0[1] + sigma*quad[j], x0[2] + sigma*quad[k]};

          const PetscInt interpCell = FindCell(cellDM, dim, x, nCells, cellList, NULL);

          if (interpCell < 0) {
            int rank;
            MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
            throw std::runtime_error("BuildInterpCellList could not determine the location of (" + std::to_string(x[0]) + ", " + std::to_string(x[1]) + ", " + std::to_string(x[2]) + ") on rank " + std::to_string(rank) + ".");
          }

          interpCellList[c*nStencil + nQuad*(k*nQuad + j) + i] = interpCell;
        }
      }
    }

    DMPlexRestoreNeighbors(cellDM, cell, 2, -1.0, -1, PETSC_FALSE, PETSC_FALSE, &nCells, &cellList) >> ablate::utilities::PetscUtilities::checkError;

  }

}


static void Reconstruction_CopyDM(DM oldDM, const PetscInt pStart, const PetscInt pEnd, const PetscInt nDOF, DM *newDM) {

  PetscSection section;


  // Create a sub auxDM

  DM coordDM;
  DMGetCoordinateDM(oldDM, &coordDM) >> ablate::utilities::PetscUtilities::checkError;

  DMClone(oldDM, newDM) >> ablate::utilities::PetscUtilities::checkError;

  // this is a hard coded "dmAux" that petsc looks for
  DMSetCoordinateDM(*newDM, coordDM) >> ablate::utilities::PetscUtilities::checkError;

  PetscSectionCreate(PetscObjectComm((PetscObject)(*newDM)), &section) >> ablate::utilities::PetscUtilities::checkError;
  PetscSectionSetChart(section, pStart, pEnd) >> ablate::utilities::PetscUtilities::checkError;
  for (PetscInt p = pStart; p < pEnd; ++p) PetscSectionSetDof(section, p, nDOF) >> ablate::utilities::PetscUtilities::checkError;
  PetscSectionSetUp(section) >> ablate::utilities::PetscUtilities::checkError;
  DMSetLocalSection(*newDM, section) >> ablate::utilities::PetscUtilities::checkError;
  PetscSectionDestroy(&section) >> ablate::utilities::PetscUtilities::checkError;
  DMSetUp(*newDM) >> ablate::utilities::PetscUtilities::checkError;

  // This builds the global section information based on the local section. It's necessary if we don't create a global vector
  //    righ away.
  DMGetGlobalSection(*newDM, &section) >> ablate::utilities::PetscUtilities::checkError;

  /* Calling DMPlexComputeGeometryFVM() generates the value returned by DMPlexGetMinRadius() */
  Vec cellgeom = NULL;
  Vec facegeom = NULL;
  DMPlexComputeGeometryFVM(*newDM, &cellgeom, &facegeom);
  VecDestroy(&cellgeom);
  VecDestroy(&facegeom);

}



// The region should be the region WITHOUT ghost cells
Reconstruction::Reconstruction(const std::shared_ptr<ablate::domain::SubDomain> subDomain, std::shared_ptr<domain::Region> region) : region(region), subDomain(subDomain) {

//  int rank;
//  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  PetscReal       h = 0.0;
  const PetscInt  dim = subDomain->GetDimensions();   // VOF and LS subdomains must have the same dimension. Can't think of a reason they wouldn't.
  const PetscInt  polyAug = 2; // Looks like I need an odd augmented polynomial order for the curvature to be acceptable
  const bool      doesNotHaveDerivatives = false;
  const bool      doesNotHaveInterpolation = false;

  DMPlexGetMinRadius(subDomain->GetDM(), &h) >> ablate::utilities::PetscUtilities::checkError;
  h *= 2.0; // Min r

  // Setup the RBF interpolants
  vertRBF = std::make_shared<ablate::domain::rbf::IMQ>(polyAug, 1e-2*h, doesNotHaveDerivatives, doesNotHaveInterpolation, true);
  vertRBF->Setup(subDomain);
  vertRBF->Initialize();


  cellRBF = std::make_shared<ablate::domain::rbf::MQ>(polyAug, h, doesNotHaveDerivatives, doesNotHaveInterpolation, false);
  cellRBF->Setup(subDomain);
  cellRBF->Initialize();

  DM subAuxDM = subDomain->GetSubAuxDM();

  // Create the ranges <--- These might be deleted if they aren't actually needed
  subDomain->GetRange(nullptr, 0, vertRange);
  subDomain->GetCellRange(region, cellRange);   // Range of cells without boundary ghosts


  // Get the point->index mapping for cells
  reverseVertRange = ablate::domain::ReverseRange(vertRange);
  reverseCellRange = ablate::domain::ReverseRange(cellRange);


  // Create individual DMs for vertex- and cell-based data. We need a separate DM for each Vec
  //    so that we can do global<->local updates on the data that has been updated, rather than
  //    everything
  PetscInt vStart, vEnd;
  DMPlexGetDepthStratum(subAuxDM, 0, &vStart, &vEnd) >> ablate::utilities::PetscUtilities::checkError;
  Reconstruction_CopyDM(subAuxDM, vStart, vEnd, 1, &vertDM);
  DMCreateLocalVector(vertDM, &lsVec[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
  DMCreateGlobalVector(vertDM, &lsVec[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;
  VecZeroEntries(lsVec[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
  VecZeroEntries(lsVec[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;

  Reconstruction_CopyDM(subAuxDM, vStart, vEnd, dim, &vertGradDM);

  // Create a DM for vertex-based data
  PetscInt cStart, cEnd;
  DMPlexGetHeightStratum(subAuxDM, 0, &cStart, &cEnd) >> ablate::utilities::PetscUtilities::checkError;
  Reconstruction_CopyDM(subAuxDM, cStart, cEnd, 1, &cellDM);
  Reconstruction_CopyDM(subAuxDM, cStart, cEnd, dim, &cellGradDM);



  // Form the list of cells that will have calculations. The list will have local values
  //    in 0 -> nLocal-1 and ghost values from nLocal->nTotal-1

  // Get the ghost cell label
  DMLabel ghostLabel;
  DMGetLabel(cellDM, "ghost", &ghostLabel) >> utilities::PetscUtilities::checkError;

  // Get the start of any boundary ghost cells
  PetscInt boundaryCellStart;
  DMPlexGetCellTypeStratum(cellDM, DM_POLYTOPE_FV_GHOST, &boundaryCellStart, nullptr) >> utilities::PetscUtilities::checkError;


  PetscMalloc2(cEnd - cStart, &cellList, cEnd - cStart, &reverseCellList) >> ablate::utilities::PetscUtilities::checkError;
  reverseCellList -= cStart;
  for (PetscInt c = 0; c < cEnd - cStart; ++c) cellList[c] = -1;

  // First the local cells
  nLocalCell = 0;
  for (PetscInt c = cStart; c < cEnd; ++c) {

    reverseCellList[c] = c - cStart;

    // Check if it's a ghost
    PetscInt isGhost = -1;
    if (ghostLabel) {
        DMLabelGetValue(ghostLabel, c, &isGhost) >> utilities::PetscUtilities::checkError;
    }

    // See if it's owned by this rank
    PetscInt owned;
    DMPlexGetPointGlobal(cellDM, c, &owned, nullptr) >> utilities::PetscUtilities::checkError;

    if (owned >= 0 && isGhost < 0 && (boundaryCellStart < 0 || c < boundaryCellStart)) {
      cellList[nLocalCell++] = c;
    }
  }

  // Now the ghost cells
  nTotalCell = nLocalCell;
  if (boundaryCellStart >= 0) {
    for (PetscInt c = cStart; c < cEnd; ++c) {

      // Check if it's a ghost
      PetscInt isGhost = -1;
      if (ghostLabel) {
          DMLabelGetValue(ghostLabel, c, &isGhost) >> utilities::PetscUtilities::checkError;
      }

      // See if it's owned by this rank
      PetscInt owned;
      DMPlexGetPointGlobal(cellDM, c, &owned, nullptr) >> utilities::PetscUtilities::checkError;

      if (owned < 0 || isGhost > 0 || c >= boundaryCellStart) {
        cellList[nTotalCell++] = c;
      }
    }
  }



//  // Now form the list of vertices.
  PetscMalloc2(vEnd - vStart, &vertList, vEnd - vStart, &reverseVertList);
  reverseVertList -= vStart;
  for (PetscInt v = 0; v < vEnd - vStart; ++v) vertList[v] = -1;

  nLocalVert = 0;
  for (PetscInt v = vStart; v < vEnd; ++v) {

    reverseVertList[v] = v - vStart;

    // See if it's owned by this rank
    PetscInt owned;
    DMPlexGetPointGlobal(vertDM, v, &owned, nullptr) >> utilities::PetscUtilities::checkError;

    if (owned >= 0 ) {
      vertList[nLocalVert++] = v;
    }
  }

  nTotalVert = nLocalVert;
  for (PetscInt v = vStart; v < vEnd; ++v) {

    reverseVertList[v] = v - vStart;

    // See if it's owned by this rank
    PetscInt owned;
    DMPlexGetPointGlobal(vertDM, v, &owned, nullptr) >> utilities::PetscUtilities::checkError;

    if (owned < 0 ) {
      vertList[nTotalVert++] = v;
    }
  }

  // Setup the convolution stencil list
  BuildInterpCellList();

}

Reconstruction::~Reconstruction() {

xexit("");


  PetscFree(interpCellList) >> ablate::utilities::PetscUtilities::checkError;

  PetscInt cStart, cEnd;
  DMPlexGetHeightStratum(cellDM, 0, &cStart, &cEnd) >> ablate::utilities::PetscUtilities::checkError;
  reverseCellList += cStart;
  PetscFree2(cellList, reverseCellList);

  PetscInt vStart, vEnd;
  DMPlexGetDepthStratum(vertDM, 0, &vStart, &vEnd) >> ablate::utilities::PetscUtilities::checkError;
  reverseVertList += vStart;
  PetscFree2(vertList, reverseVertList);

  DMDestroy(&(vertDM)) >> ablate::utilities::PetscUtilities::checkError;
  DMDestroy(&(vertGradDM)) >> ablate::utilities::PetscUtilities::checkError;
  DMDestroy(&(cellDM)) >> ablate::utilities::PetscUtilities::checkError;
  DMDestroy(&(cellGradDM)) >> ablate::utilities::PetscUtilities::checkError;

  for (int i = 0; i < 2; ++i) {
    VecDestroy(&(lsVec[i])) >> ablate::utilities::PetscUtilities::checkError;
//    VecDestroy(&(vertGradVec[i])) >> ablate::utilities::PetscUtilities::checkError;
//    VecDestroy(&(cellVec[i])) >> ablate::utilities::PetscUtilities::checkError;
//    VecDestroy(&(cellGradVec[i])) >> ablate::utilities::PetscUtilities::checkError;
  }

  cellRBF.reset();
  vertRBF.reset();
}

void Reconstruction_SaveCellData(DM dm, const Vec vec, const char fname[255], const PetscInt id, PetscInt Nc, std::shared_ptr<ablate::domain::SubDomain> subDomain) {

  ablate::domain::Range range;
  const PetscScalar *array;
  PetscInt      dim = subDomain->GetDimensions();
  MPI_Comm      comm = PetscObjectComm((PetscObject)dm);
  int rank, size;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  subDomain->GetCellRange(nullptr, range);

  VecGetArrayRead(vec, &array) >> ablate::utilities::PetscUtilities::checkError;

  PetscInt boundaryCellStart;
  DMPlexGetCellTypeStratum(dm, DM_POLYTOPE_FV_GHOST, &boundaryCellStart, nullptr) >> ablate::utilities::PetscUtilities::checkError;


  for (PetscInt r = 0; r < size; ++r) {
    if ( rank==r ) {

      FILE *f1;
      if ( rank==0 ) f1 = fopen(fname, "w");
      else f1 = fopen(fname, "a");

      for (PetscInt c = range.start; c < range.end; ++c) {
        PetscInt cell = range.points ? range.points[c] : c;

        DMPolytopeType ct;
        DMPlexGetCellType(dm, cell, &ct) >> ablate::utilities::PetscUtilities::checkError;

        if (ct < 12) {

          PetscReal x0[3];
          DMPlexComputeCellGeometryFVM(dm, cell, NULL, x0, NULL) >> ablate::utilities::PetscUtilities::checkError;
          for (PetscInt d = 0; d < dim; ++d) {
            fprintf(f1, "%+e\t", x0[d]);
          }

          const PetscScalar *val;
          xDMPlexPointLocalRead(dm, cell, id, array, &val) >> ablate::utilities::PetscUtilities::checkError;
          for (PetscInt i = 0; i < Nc; ++i) {
            fprintf(f1, "%+e\t", val[i]);
          }

          fprintf(f1, "\n");
        }
      }
      fclose(f1);
    }

    MPI_Barrier(PETSC_COMM_WORLD);
  }


  VecRestoreArrayRead(vec, &array) >> ablate::utilities::PetscUtilities::checkError;
  ablate::domain::RestoreRange(range);
}

void Reconstruction_SaveCellData(DM dm, const Vec vec, const char fname[255], const ablate::domain::Field *field, PetscInt Nc, std::shared_ptr<ablate::domain::SubDomain> subDomain) {
  Reconstruction_SaveCellData(dm, vec, fname, field->id, Nc, subDomain);
}

void Reconstruction::SaveData(DM dm, const PetscInt *array, const PetscInt nList, const PetscInt *list, const char fname[255], PetscInt Nc) {

  int rank, size;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  MPI_Comm_size(PETSC_COMM_WORLD, &size);
  PetscInt      dim = subDomain->GetDimensions();

  for (PetscInt r = 0; r < size; ++r) {
    if ( rank==r ) {

      FILE *f1;
      if ( rank==0 ) f1 = fopen(fname, "w");
      else f1 = fopen(fname, "a");

      for (PetscInt p = 0; p < nList; ++p) {
        const PetscInt point = list[p];
        PetscReal x[3];
        DMPlexComputeCellGeometryFVM(dm, point, NULL, x, NULL) >> ablate::utilities::PetscUtilities::checkError;

        for (PetscInt d = 0; d < dim; ++d) fprintf(f1, "%.16e\t", x[d]);

        for (PetscInt d = 0; d < Nc; ++d) fprintf(f1, "%ld\t", array[p*Nc + d]);

        fprintf(f1, "\n");
      }
      fclose(f1);
    }

    MPI_Barrier(PETSC_COMM_WORLD);
  }
}

void Reconstruction::SaveData(DM dm, const PetscScalar *array, const PetscInt nList, const PetscInt *list, const char fname[255], PetscInt Nc) {

  int rank, size;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  MPI_Comm_size(PETSC_COMM_WORLD, &size);
  PetscInt      dim = subDomain->GetDimensions();

  for (PetscInt r = 0; r < size; ++r) {
    if ( rank==r ) {

      FILE *f1;
      if ( rank==0 ) f1 = fopen(fname, "w");
      else f1 = fopen(fname, "a");

      for (PetscInt p = 0; p < nList; ++p) {
        const PetscInt point = list[p];
        PetscReal x[3];
        DMPlexComputeCellGeometryFVM(dm, point, NULL, x, NULL) >> ablate::utilities::PetscUtilities::checkError;

        for (PetscInt d = 0; d < dim; ++d) fprintf(f1, "%.16e\t", x[d]);

        for (PetscInt d = 0; d < Nc; ++d) fprintf(f1, "%.16e\t", array[p*Nc + d]);

        fprintf(f1, "\n");
      }
      fclose(f1);
    }

    MPI_Barrier(PETSC_COMM_WORLD);
  }
}

void Reconstruction::SaveData(DM dm, const Vec vec, const PetscInt nList, const PetscInt *list, const char fname[255], PetscInt Nc) {
  const PetscScalar *array;
  VecGetArrayRead(vec, &array) >> ablate::utilities::PetscUtilities::checkError;
  SaveData(dm, array, nList, list, fname, Nc);
  VecRestoreArrayRead(vec, &array) >> ablate::utilities::PetscUtilities::checkError;
}

// Set the cell masks and the vof gradient.
//
// This has to be done very carefully. Consider a 1D mesh: 1 - 2 - 3 - 4 - 5 - 6.
//    Cells 1 to 3 are on P0 and cells 4 to 6 are on P1. Have one cell overlap, thus
//    on P0: 1 - 2 - 3 - G4. Let cell-3 be the only cut-cell. We need to be able to mark cell-4 as the next level.
//    One possible solution is to increase the overlap and then march over all cells (not just local ones). Problem with that is
//    I don't want to depend on the user creating a YAML file with the required size of overlap just to accomodate this function,
//    which would (probably) slow down large-scale computations. It might be possible to use a different overlap for
//    cellDM, etc, but A) some time would need to be spent seeing how to do this and B) how beneficial would it be for the
//    rest of the reconstruction?
//
//  Instead we'll use an accumulator approach where new cells are marked. An ADD_VALUES operation is then done and
//    new cells at that level are added to a temporary array. This SHOULD only require one-level of overlap
void Reconstruction::SetMasksAndNormal(const PetscInt nLevels, PetscInt *cellMask, PetscInt *vertMask, Vec vofVec[2], Vec vofGradVec[2]) {

  PetscScalar *vofArray = nullptr, *cellMaskVecArray = nullptr, *vofGradArray = nullptr;
  Vec cellMaskVec[2] = {nullptr, nullptr};



  DMGetLocalVector(cellDM, &cellMaskVec[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
  DMGetGlobalVector(cellDM, &cellMaskVec[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;

  VecGetArray(vofVec[LOCAL], &vofArray) >> ablate::utilities::PetscUtilities::checkError;
  VecGetArray(vofGradVec[LOCAL], &vofGradArray) >> ablate::utilities::PetscUtilities::checkError;
  VecGetArray(cellMaskVec[LOCAL], &cellMaskVecArray);
  for (PetscInt c = 0; c < nTotalCell; ++c) {

    if ((vofArray[c] > 0.001) && (vofArray[c] < 0.999)) {
      cellMaskVecArray[c] = cellMask[c] =  1;


      const PetscInt cell = cellList[c];

      PetscScalar *n;
      DMPlexPointLocalRef(cellGradDM, cell, vofGradArray, &n) >> ablate::utilities::PetscUtilities::checkError;

      DMPlexCellGradFromCell(cellDM, cell, vofVec[LOCAL], -1, 0, n) >> ablate::utilities::PetscUtilities::checkError;
    }
    else {
      cellMask[c] = 0;
    }
  }


//   Turn off any "cut cells" where the cell is not surrounded by any other cut cells.
//   To avoid cut-cells two cells-thick turn off any cut-cells which have a neighoring gradient passing through them.
  const PetscInt    dim = subDomain->GetDimensions();
  for (PetscInt c = 0; c < nLocalCell; ++c) {

    if (cellMask[c] == 1) {

      const PetscInt cell = cellList[c];

      PetscInt nCells, *cells;
      DMPlexGetNeighbors(cellDM, cell, 1, -1.0, -1, PETSC_FALSE, PETSC_FALSE, &nCells, &cells) >> ablate::utilities::PetscUtilities::checkError;
      PetscInt nCut = 0;
      for (PetscInt i = 0; i < nCells; ++i) {
        PetscInt id = reverseCellList[cells[i]];
        nCut += (cellMaskVecArray[id] > 0.5);
      }
      DMPlexRestoreNeighbors(cellDM, cell, 1, -1.0, -1, PETSC_FALSE, PETSC_FALSE, &nCells, &cells) >> ablate::utilities::PetscUtilities::checkError;

      cellMask[c] = (nCut==1 ? 0 : 1); // If nCut equals 1 then the center cell is the only cut cell, so temporarily mark it as 1.


      const PetscScalar *n = nullptr;
      DMPlexPointLocalRead(cellGradDM, cell, vofGradArray, &n) >> ablate::utilities::PetscUtilities::checkError; // VOF normal

      if (cellMask[c]==1 && ablate::utilities::MathUtilities::MagVector(dim, n)>PETSC_SMALL) {
        // Now check for two-deep cut-cells.
        const PetscReal direction[2] = {-1.0, +1.0};
        for (PetscInt d = 0; d < 2; ++d) {
          PetscInt neighborCell = -1;
          DMPlexGetForwardCell(cellDM, cell, n, direction[d], &neighborCell) >> ablate::utilities::PetscUtilities::checkError;
          if (neighborCell > -1) {
            neighborCell = reverseCellList[neighborCell];

            if (PetscAbsReal(vofArray[neighborCell] - 0.5) < PetscAbsReal(vofArray[c] - 0.5)) {
              cellMask[c] = 0;
              break;
            }
          }
        }
      }
    }
  }

  VecRestoreArray(vofVec[LOCAL], &vofArray) >> ablate::utilities::PetscUtilities::checkError;
  VecRestoreArray(vofGradVec[LOCAL], &vofGradArray) >> ablate::utilities::PetscUtilities::checkError;
  VecRestoreArray(cellMaskVec[LOCAL], &cellMaskVecArray) >> ablate::utilities::PetscUtilities::checkError;


  // Now label the surrounding cells
  for (PetscInt l = 1; l < nLevels; ++l) {

    VecZeroEntries(cellMaskVec[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
    VecGetArray(cellMaskVec[LOCAL], &cellMaskVecArray) >> ablate::utilities::PetscUtilities::checkError;
    for (PetscInt c = 0; c < nLocalCell; ++c) {
      if ( cellMask[c] == l ) {
        PetscInt cell = cellList[c];
        PetscInt nCells, *cells;
        DMPlexGetNeighbors(cellDM, cell, 1, -1.0, -1, PETSC_FALSE, PETSC_FALSE, &nCells, &cells) >> ablate::utilities::PetscUtilities::checkError;
        for (PetscInt i = 0; i < nCells; ++i) {
          PetscScalar *neighborMaskVal = nullptr;
          DMPlexPointLocalRef(cellDM, cells[i], cellMaskVecArray, &neighborMaskVal) >> ablate::utilities::PetscUtilities::checkError;
          ++(*neighborMaskVal);
        }
        DMPlexRestoreNeighbors(cellDM, cell, 1, -1.0, -1, PETSC_FALSE, PETSC_FALSE, &nCells, &cells) >> ablate::utilities::PetscUtilities::checkError;
      }
    }
    VecRestoreArray(cellMaskVec[LOCAL], &cellMaskVecArray) >> ablate::utilities::PetscUtilities::checkError;

    VecZeroEntries(cellMaskVec[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;
    DMLocalToGlobal(cellDM, cellMaskVec[LOCAL], ADD_VALUES, cellMaskVec[GLOBAL]) >> utilities::PetscUtilities::checkError;
    DMGlobalToLocal(cellDM, cellMaskVec[GLOBAL], INSERT_VALUES, cellMaskVec[LOCAL]) >> utilities::PetscUtilities::checkError;

    VecGetArray(cellMaskVec[LOCAL], &cellMaskVecArray) >> ablate::utilities::PetscUtilities::checkError;
    for (PetscInt c = 0; c < nTotalCell; ++c) {
      if (cellMask[c] == 0 && cellMaskVecArray[c]>0.5) cellMask[c] = l + 1;
    }
    VecRestoreArray(cellMaskVec[LOCAL], &cellMaskVecArray) >> ablate::utilities::PetscUtilities::checkError;

  }

  // Set the vertex mask
  for (PetscInt v = 0; v < nTotalVert; ++v) vertMask[v] = nLevels + 2;

  for (PetscInt c = 0; c < nTotalCell; ++c) {

    if (cellMask[c] > 0) {

      const PetscInt cell = cellList[c];

      PetscInt nVert, *verts;
      DMPlexCellGetVertices(vertDM, cell, &nVert, &verts) >> ablate::utilities::PetscUtilities::checkError;

      for (PetscInt v = 0; v < nVert; ++v) {
        const PetscInt id = reverseVertList[verts[v]];
        vertMask[id] = PetscMin(vertMask[id], cellMask[c]);
      }
      DMPlexCellRestoreVertices(vertDM, cell, &nVert, &verts) >> ablate::utilities::PetscUtilities::checkError;
    }
  }

  // Switch all deactivated vertices to -1
  for (PetscInt v = 0; v < nTotalVert; ++v) {
    vertMask[v] = (vertMask[v]==(nLevels+2)) ? 0 : vertMask[v];
  }

  DMRestoreLocalVector(cellDM, &cellMaskVec[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
  DMRestoreGlobalVector(cellDM, &cellMaskVec[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;

}

// vofVec MUST have ghost cell information
void Reconstruction::CutCellLevelSetValues(Vec vofVec, const PetscInt *cellMask, const PetscInt *vertMask, Vec lsVec[2]) {

  MPI_Comm lsCOMM = PetscObjectComm((PetscObject)vertDM);

  const PetscInt    dim = subDomain->GetDimensions();   // VOF and LS subdomains must have the same dimension. Can't think of a reason they wouldn't.

  // First get the number of cut-cells associated with each vertex
  PetscInt *lsCount = nullptr;
  DMGetWorkArray(vertDM, nLocalVert, MPIU_INT, &lsCount) >> ablate::utilities::PetscUtilities::checkError;
  for (PetscInt v = 0; v < nTotalVert; ++v) {

    lsCount[v] = 0;
    if (vertMask[v]==1) {
      const PetscInt vert = vertList[v];

      PetscInt nc, *cells;
      DMPlexVertexGetCells(vertDM, vert, &nc, &cells) >> ablate::utilities::PetscUtilities::checkError;

      for (PetscInt c = 0; c < nc; ++c){
        const PetscInt id = reverseCellList[cells[c]];
        lsCount[v] += (cellMask[id] == 1);
      }

      if (lsCount[v] < 1) {
        PetscReal x[dim];
        DMPlexComputeCellGeometryFVM(vertDM, vert, NULL, x, NULL) >> ablate::utilities::PetscUtilities::checkError;
        printf("%ld;plot(%f,%f,'r*');\n", v, x[0], x[1]);
        throw std::runtime_error("Vertex is marked as next to a cut cell but is not!");
      }

      DMPlexVertexRestoreCells(vertDM, vert, &nc, &cells) >> ablate::utilities::PetscUtilities::checkError;
    }

  }

  // Approximate the unit normal at the cell center using the VOF data
  PetscReal *cellGrad = nullptr;
  DMGetWorkArray(cellGradDM, nLocalCell, MPIU_REAL, &cellGrad) >> ablate::utilities::PetscUtilities::checkError;
  for (PetscInt c = 0; c < nLocalCell; ++c) {
    if (cellMask[c] == 1) {
      DMPlexCellGradFromCell(cellDM, cellList[c], vofVec, -1, 0, &cellGrad[c*dim]) >> ablate::utilities::PetscUtilities::checkError;
      ablate::utilities::MathUtilities::NormVector(dim, &cellGrad[c*dim]);
      ablate::utilities::MathUtilities::ScaleVector(dim, &cellGrad[c*dim], -1.0);
    }
  }


  PetscReal         h = 0.0;
  DMPlexGetMinRadius(vertDM, &h) >> ablate::utilities::PetscUtilities::checkError;
  h *= 2.0; // Min radius returns the distance between a cell-center and a face. Double it to get the average cell size
  PetscReal maxDiff = 10*h;
  PetscInt iter = 0;


  const PetscScalar *vofArray = nullptr;
  VecGetArrayRead(vofVec, &vofArray) >> ablate::utilities::PetscUtilities::checkError;

  while ( maxDiff > 1e-3*h && iter<200 ) {

    ++iter;

    PetscScalar *lsArray[2] = {nullptr, nullptr};

    VecZeroEntries(lsVec[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;
    VecGetArray(lsVec[GLOBAL], &lsArray[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;
    for (PetscInt c = 0; c < nLocalCell; ++c) {

      // Only worry about cut-cells
      if ( cellMask[c] == 1 ) {
        PetscInt cell = cellList[c];

        // The VOF for the cell
        const PetscScalar *vofVal = nullptr;
        DMPlexPointLocalRead(cellDM, cell, vofArray, &vofVal) >> ablate::utilities::PetscUtilities::checkError;

        PetscInt nv, *verts;
        DMPlexCellGetVertices(vertDM, cell, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;

        PetscReal *lsVertVals = NULL;
        DMGetWorkArray(vertDM, nv, MPIU_REAL, &lsVertVals) >> ablate::utilities::PetscUtilities::checkError;

        // Level set values at the vertices
        ablate::levelSet::Utilities::VertexLevelSet_VOF(vertDM, cell, *vofVal, &cellGrad[c*dim], &lsVertVals);

//{
//  PetscReal x[dim];
//  DMPlexComputeCellGeometryFVM(vertDM, cell, NULL, x, NULL) >> ablate::utilities::PetscUtilities::checkError;
//  printf("Reconstruction\n");
//  printf("Cell %ld: %+f\t%+f\n", cell, x[0], x[1]);
//  for (PetscInt v = 0; v < nv; ++v) {
//    PetscReal x[dim];
//    DMPlexComputeCellGeometryFVM(vertDM, verts[v], NULL, x, NULL) >> ablate::utilities::PetscUtilities::checkError;
//    printf("Vert %ld: %+f\t%+f\t%+e\n", verts[v], x[0], x[1], lsVertVals[v]);
//  }
//  xexit("");

//}

        for (PetscInt v = 0; v < nv; ++v) {
          const PetscInt id = reverseVertList[verts[v]];
          lsArray[GLOBAL][id] += lsVertVals[v];
        }

        DMRestoreWorkArray(vertDM, nv, MPIU_REAL, &lsVertVals) >> ablate::utilities::PetscUtilities::checkError;
        DMPlexCellRestoreVertices(vertDM, cell, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;
      }
    }

    VecGetArray(lsVec[LOCAL], &lsArray[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
    maxDiff = -1.0;
//FILE *f1 = fopen("Reconstruction.txt","w");
    for (PetscInt v = 0; v < nLocalVert; ++v) {
      if (vertMask[v] == 1) {
        lsArray[GLOBAL][v] /= lsCount[v];
//PetscReal x[dim];
//DMPlexComputeCellGeometryFVM(vertDM, vertList[v], NULL, x, NULL) >> ablate::utilities::PetscUtilities::checkError;
//fprintf(f1, "%f\t%f\t%ld\t%+e\n", x[0], x[1], lsCount[v],lsArray[GLOBAL][v]);


        maxDiff = PetscMax(maxDiff, PetscAbsReal(lsArray[GLOBAL][v] - lsArray[LOCAL][v]));

      }
    }
//fclose(f1);
//xexit("");
    // Get the maximum change across all processors. This also acts as a sync point
    MPI_Allreduce(MPI_IN_PLACE, &maxDiff, 1, MPIU_REAL, MPIU_MAX, lsCOMM);

    VecRestoreArray(lsVec[GLOBAL], &lsArray[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;
    VecRestoreArray(lsVec[LOCAL], &lsArray[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;

    // It is necessary to communicate the updates, otherwise errors at the edge of ghost cells will
    //    propogate through the domain.
    DMGlobalToLocal(vertDM, lsVec[GLOBAL], INSERT_VALUES, lsVec[LOCAL]) >> utilities::PetscUtilities::checkError;

    // Update the cell-center normal using the level-set data
    for (PetscInt c = 0; c < nLocalCell; ++c) {
      if (cellMask[c] == 1) {
        DMPlexCellGradFromVertex(vertDM, cellList[c], lsVec[LOCAL], -1, 0, &cellGrad[c*dim]) >> ablate::utilities::PetscUtilities::checkError;
        ablate::utilities::MathUtilities::NormVector(dim, &cellGrad[c*dim]);
      }
    }

    PetscPrintf(PETSC_COMM_WORLD, "Cut Cells %" PetscInt_FMT": %+e\n", iter, maxDiff) >> ablate::utilities::PetscUtilities::checkError;

  }

  VecRestoreArrayRead(vofVec, &vofArray) >> ablate::utilities::PetscUtilities::checkError;
  DMRestoreWorkArray(vertDM, nLocalVert, MPIU_INT, &lsCount) >> ablate::utilities::PetscUtilities::checkError;
  DMRestoreWorkArray(cellGradDM, nLocalCell, MPIU_REAL, &cellGrad) >> ablate::utilities::PetscUtilities::checkError;

SaveData(vertDM, lsVec[LOCAL], nTotalVert, vertList, "vertLS_L.txt", 1);
SaveData(vertDM, lsVec[GLOBAL], nLocalVert, vertList, "vertLS_G.txt", 1);

//xexit("");

//  if (maxDiff > 1e-3*h) {
//    throw std::runtime_error("Interface reconstruction has failed.\n");
//  }

}

void Reconstruction::SmoothVOF(DM vofDM, Vec vofVec, const PetscInt vofID, DM smoothVOFDM, Vec smoothVOFVec[2], const PetscInt* subpointIndices) {


  // Smooth out the VOF field by averaging to vertices and then averaging to cell-centers
  const PetscScalar *vofArray = nullptr;
  VecGetArrayRead(vofVec, &vofArray) >> ablate::utilities::PetscUtilities::checkError;

  PetscReal *lsVertVals = NULL;
  DMGetWorkArray(vertDM, nTotalVert, MPIU_REAL, &lsVertVals) >> ablate::utilities::PetscUtilities::checkError;

  // This is done over local and ghost vertices so that communication isn't necessary
  for (PetscInt v = 0; v < nTotalVert; ++v) {
    const PetscInt vert = vertList[v];

    PetscInt nCells, *cellList;
    DMPlexVertexGetCells(vertDM, vert, &nCells, &cellList) >> ablate::utilities::PetscUtilities::checkError;

    PetscScalar *smoothVOF;
    DMPlexPointLocalRef(vertDM, vert, lsVertVals, &smoothVOF) >> ablate::utilities::PetscUtilities::checkError;
    *smoothVOF = 0.0;

    for (PetscInt i = 0; i < nCells; ++i) {
      const PetscInt globalCell = subpointIndices ? subpointIndices[cellList[i]] : cellList[i];
      const PetscScalar *vof = nullptr;
      xDMPlexPointLocalRead(vofDM, globalCell, vofID, vofArray, &vof) >> ablate::utilities::PetscUtilities::checkError;
      *smoothVOF += *vof;
    }
    *smoothVOF /= nCells;

    DMPlexVertexRestoreCells(vertDM, vert, &nCells, &cellList) >> ablate::utilities::PetscUtilities::checkError;
  }
  VecRestoreArrayRead(vofVec, &vofArray) >> ablate::utilities::PetscUtilities::checkError;

  SaveData(vertDM, lsVertVals, nLocalVert, vertList, "vertVOF.txt", 1);


  PetscScalar  *smoothVOFArray = nullptr;//, *cellGradArray = nullptr;
  VecGetArray(smoothVOFVec[GLOBAL], &smoothVOFArray) >> ablate::utilities::PetscUtilities::checkError;

  // This is done only over local cells as the ghost cells at the boundary will have incorrect
  //  values due to not having all of the information.
  for (PetscInt c = 0; c < nLocalCell; ++c){
    const PetscInt cell = cellList[c];

    PetscInt nVert, *vertList;
    DMPlexCellGetVertices(cellDM, cell, &nVert, &vertList) >> ablate::utilities::PetscUtilities::checkError;

    PetscScalar *cellVOF;
    DMPlexPointLocalRef(cellDM, cell, smoothVOFArray, &cellVOF) >> ablate::utilities::PetscUtilities::checkError;
    *cellVOF = 0.0;

    for (PetscInt i = 0; i < nVert; ++i) {
      const PetscScalar *vof;
      DMPlexPointLocalRef(vertDM, vertList[i], lsVertVals, &vof) >> ablate::utilities::PetscUtilities::checkError;
      *cellVOF += *vof;
    }
    *cellVOF /= nVert;

    DMPlexCellRestoreVertices(cellDM, cell, &nVert, &vertList) >> ablate::utilities::PetscUtilities::checkError;
  }
  DMRestoreWorkArray(vertDM, nTotalVert, MPIU_REAL, &lsVertVals) >> ablate::utilities::PetscUtilities::checkError;
  VecRestoreArray(smoothVOFVec[GLOBAL], &smoothVOFArray) >> ablate::utilities::PetscUtilities::checkError;
  DMGlobalToLocal(cellDM, smoothVOFVec[GLOBAL], INSERT_VALUES, smoothVOFVec[LOCAL]) >> utilities::PetscUtilities::checkError;

}

void Reconstruction::ToLevelSet(DM vofDM, Vec vofVec, const ablate::domain::Field vofField) {

int rank, size;
MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
MPI_Comm_size(PETSC_COMM_WORLD, &size);


  PetscReal         h = 0.0;

  // Only needed if this is defined over a sub-region of the DM
  IS subpointIS;
  const PetscInt* subpointIndices = nullptr;
  if (subDomain->GetSubAuxDM()!=subDomain->GetAuxDM()) {
    DMPlexGetSubpointIS(subDomain->GetSubAuxDM(), &subpointIS) >> utilities::PetscUtilities::checkError;
    ISGetIndices(subpointIS, &subpointIndices) >> utilities::PetscUtilities::checkError;
  }

  DMPlexGetMinRadius(vofDM, &h) >> ablate::utilities::PetscUtilities::checkError;
  h *= 2.0; // Min radius returns the distance between a cell-center and a face. Double it to get the average cell size


  PetscInt cStart = -1, cEnd = -1;
  DMPlexGetHeightStratum(cellDM, 0, &cStart, &cEnd) >> ablate::utilities::PetscUtilities::checkError;

  PetscInt vStart = -1, vEnd = -1;
  DMPlexGetDepthStratum(vertDM, 0, &vStart, &vEnd) >> ablate::utilities::PetscUtilities::checkError;

/******** Smooth the VOF field ******************************************************************************/
  Vec smoothVOFVec[2];
  DMGetLocalVector(cellDM, &smoothVOFVec[LOCAL]);
  DMGetGlobalVector(cellDM, &smoothVOFVec[GLOBAL]);
//  SmoothVOF(vofDM, vofVec, vofField.id, cellDM, smoothVOFVec, subpointIndices);

{
  const PetscScalar *vofArray = nullptr;
  PetscScalar *smoothVOFArray = nullptr;
  VecGetArrayRead(vofVec, &vofArray) >> ablate::utilities::PetscUtilities::checkError;
  VecGetArray(smoothVOFVec[GLOBAL], &smoothVOFArray) >> ablate::utilities::PetscUtilities::checkError;
  for (PetscInt c = 0; c < nLocalCell; ++c) {
    const PetscInt cell = cellList[c];

    const PetscInt globalCell = subpointIndices ? subpointIndices[cell] : cell;
    const PetscScalar *vof = nullptr;
    xDMPlexPointLocalRead(vofDM, globalCell, vofField.id, vofArray, &vof) >> ablate::utilities::PetscUtilities::checkError;

    smoothVOFArray[c] = *vof;
  }
  VecRestoreArrayRead(vofVec, &vofArray) >> ablate::utilities::PetscUtilities::checkError;
  VecRestoreArray(smoothVOFVec[GLOBAL], &smoothVOFArray) >> ablate::utilities::PetscUtilities::checkError;
  DMGlobalToLocal(cellDM, smoothVOFVec[GLOBAL], INSERT_VALUES, smoothVOFVec[LOCAL]) >> utilities::PetscUtilities::checkError;

}



/**************** Determine the cut-cells and the initial cell-normal  *************************************/
//
// NOTE: I need to check if using the smoothed VOF or the true VOF give better results

  Vec vofGradVec[2] = {nullptr, nullptr};
  DMGetLocalVector(cellGradDM, &vofGradVec[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
  DMGetGlobalVector(cellGradDM, &vofGradVec[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;

  PetscInt *vertMask = nullptr, *cellMask = nullptr;
  DMGetWorkArray(vertDM, nTotalVert, MPIU_INT, &vertMask) >> ablate::utilities::PetscUtilities::checkError;
  DMGetWorkArray(cellDM, nTotalCell, MPIU_INT, &cellMask) >> ablate::utilities::PetscUtilities::checkError;


  SetMasksAndNormal(3, cellMask, vertMask, smoothVOFVec, vofGradVec);

SaveData(cellDM, cellMask, nTotalCell, cellList, "cellMask.txt", 1);
SaveData(vertDM, vertMask, nTotalVert, vertList, "vertMask.txt", 1);

  CutCellLevelSetValues(smoothVOFVec[LOCAL], cellMask, vertMask, lsVec);



SaveData(vertDM, lsVec[LOCAL], nTotalVert, vertList, "vertLS_L.txt", 1);
SaveData(vertDM, lsVec[GLOBAL], nLocalVert, vertList, "vertLS_G.txt", 1);

  DMRestoreWorkArray(vertDM, nTotalVert, MPIU_INT, &vertMask) >> ablate::utilities::PetscUtilities::checkError;
//  DMRestoreLocalVector(cellGradDM, &cellGradVec[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
//  DMRestoreGlobalVector(cellGradDM, &cellGradVec[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;

  DMRestoreLocalVector(cellDM, &smoothVOFVec[LOCAL]);
  DMRestoreGlobalVector(cellDM, &smoothVOFVec[GLOBAL]);





///**************** Iterate to get the level-set values at vertices *************************************/

//  // Temporary level-set work array to store old values
//  PetscScalar *tempLS;
//  DMGetWorkArray(auxDM, vertRange.end - vertRange.start, MPIU_SCALAR, &tempLS) >> ablate::utilities::PetscUtilities::checkError;
//  tempLS -= vertRange.start;

//  PetscReal maxDiff = 1.0;
//  PetscInt iter = 0;

//  MPI_Comm auxCOMM = PetscObjectComm((PetscObject)auxDM);

////SaveCellData(auxDM, auxVec, "normal0.txt", cellNormalField, dim, subDomain);


//  while ( maxDiff > 1e-3*h && iter<100 ) {

//    ++iter;

//    for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {
//      if (vertMask[v]==1) {
//        PetscInt vert = vertRange.GetPoint(v);
//        const PetscReal *oldLS = nullptr;
//        xDMPlexPointLocalRead(auxDM, vert, lsID, auxArray, &oldLS) >> ablate::utilities::PetscUtilities::checkError;
//        tempLS[v] = *oldLS;
//      }
//    }

//    // Note: The unit normal and CutCellLevelSetValues must work on the same set of datat.

//    // This updates the lsField by taking the average vertex values necessary to match the VOF in cutcells
//    CutCellLevelSetValues(subDomain, cellRange, vertRange, reverseVertRange, cellMask, cutCellDM, cutCellVec, vofID, auxDM, auxVec, cellNormalID, lsID);

//    //     Update the normals
//    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
//      if (cellMask[c] == 1) {
//        PetscInt cell = cellRange.GetPoint(c);
//        PetscScalar *n = nullptr;
//        xDMPlexPointLocalRef(auxDM, cell, cellNormalID, auxArray, &n);
//        DMPlexCellGradFromVertex(auxDM, cell, auxVec, lsID, 0, n) >> ablate::utilities::PetscUtilities::checkError;
//        ablate::utilities::MathUtilities::NormVector(dim, n);
//      }
//    }

//    subDomain->UpdateAuxLocalVector();


//    // Now compute the difference on this processor
//    maxDiff = -1.0;
//    for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {

//      if (vertMask[v] == 1) {
//        PetscInt vert = vertRange.GetPoint(v);
//        const PetscReal *newLS = nullptr;
//        xDMPlexPointLocalRead(auxDM, vert, lsID, auxArray, &newLS) >> ablate::utilities::PetscUtilities::checkError;

//        maxDiff = PetscMax(maxDiff, PetscAbsReal(tempLS[v] - *newLS));
//      }
//    }
//    // Get the maximum change across all processors. This also acts as a sync point
//    MPI_Allreduce(MPI_IN_PLACE, &maxDiff, 1, MPIU_REAL, MPIU_MAX, auxCOMM);
//#ifdef saveData
//    PetscPrintf(PETSC_COMM_WORLD, "Cut Cells %" PetscInt_FMT": %+e\n", iter, maxDiff) >> ablate::utilities::PetscUtilities::checkError;
//#endif
//  }

//  if (maxDiff > 1e-3*h) {
//    throw std::runtime_error("Interface reconstruction has failed.\n");
//  }


//#ifdef saveData
//  sprintf(fname, "ls1_%03ld.txt", saveIter);
//  SaveVertexData(auxDM, auxVec, fname, lsField, 1, subDomain);
//#endif


///**************** Set the data in the rest of the domain to be a large value *************************************/
////PetscPrintf(PETSC_COMM_WORLD, "Setting data\n");
//  // Set the vertices far away as the largest possible value in the domain with the appropriate sign.
//  // This is done after the determination of cut-cells so that all vertices associated with cut-cells have been marked.
//  PetscReal gMin[3], gMax[3], maxDist = -1.0;
//  DMGetBoundingBox(auxDM, gMin, gMax) >> ablate::utilities::PetscUtilities::checkError;

//  for (PetscInt d = 0; d < dim; ++d) {
//    maxDist = PetscMax(maxDist, gMax[d] - gMin[d]);
//  }
//  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
//    PetscInt cell = cellRange.GetPoint(c);

//    // Only worry about cells to far away
//    if ( cellMask[c] == 0 && ablate::levelSet::Utilities::ValidCell(solDM, cell)) {
//      const PetscScalar *vofVal = nullptr;
//      xDMPlexPointLocalRead(solDM, cell, vofID, solArray, &vofVal) >> ablate::utilities::PetscUtilities::checkError;

//      PetscReal sgn = PetscSignReal(0.5 - (*vofVal));

//      PetscInt nv, *verts;
//      DMPlexCellGetVertices(solDM, cell, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;

//      for (PetscInt v = 0; v < nv; ++v) {
//        PetscInt id = reverseVertRange.GetIndex(verts[v]);
//        if (vertMask[id] == 0) {
//          PetscScalar *lsVal = nullptr;
//          xDMPlexPointLocalRef(auxDM, verts[v], lsID, auxArray, &lsVal) >> ablate::utilities::PetscUtilities::checkError;
//          *lsVal = sgn*maxDist;
//        }
//      }
//      DMPlexCellRestoreVertices(solDM, cell, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;
//    }
//  }



///**************** Mark the cells that need to be udpated via the reinitialization equation *************************************/
////PetscPrintf(PETSC_COMM_WORLD, "Marking cells\n");
//  // Mark all of the cells neighboring cells level-by-level.
//  // Note that DMPlexGetNeighbors has an issue in parallel whereby cells will be missed due to the unknown partitioning -- Need to investigate

//  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
//    PetscInt cell = cellRange.GetPoint(c);
//    PetscScalar *maskVal = nullptr;
//    xDMPlexPointLocalRef(auxDM, cell, vofID, workArray, &maskVal) >> ablate::utilities::PetscUtilities::checkError;
//    *maskVal = cellMask[c];
//  }


//  DMLocalToGlobal(auxDM, workVec, INSERT_VALUES, workVecGlobal) >> utilities::PetscUtilities::checkError;
//  DMGlobalToLocal(auxDM, workVecGlobal, INSERT_VALUES, workVec) >> utilities::PetscUtilities::checkError;


//  PetscReal lsRange[2] = {PETSC_MAX_REAL, -PETSC_MAX_REAL};

//  for (PetscInt v = vertRange.start; v < vertRange.end; ++v){
//    if (vertMask[v]==1) {
//      PetscInt vert = vertRange.GetPoint(v);
//      const PetscScalar *lsVal;
//      xDMPlexPointLocalRead(auxDM, vert, lsID, auxArray, &lsVal) >> ablate::utilities::PetscUtilities::checkError;
//      lsRange[0] = PetscMin(lsRange[0], *lsVal);
//      lsRange[1] = PetscMax(lsRange[1], *lsVal);
//    }
//  }


//  lsRange[0] = -lsRange[0];
//  MPI_Allreduce(MPI_IN_PLACE, lsRange, 2, MPIU_REAL, MPIU_MAX, auxCOMM);
//  lsRange[0] = -lsRange[0];




//  for (PetscInt l = 1; l <= nLevels; ++l) {


//    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
//      PetscInt cell = cellRange.GetPoint(c);
//      PetscScalar *maskVal = nullptr;
//      xDMPlexPointLocalRef(auxDM, cell, vofID, workArray, &maskVal) >> ablate::utilities::PetscUtilities::checkError;

//      if ( PetscAbsScalar(*maskVal - l) < 0.1 ) {
//        PetscInt nCells, *cells;
//        DMPlexGetNeighbors(solDM, cell, 1, -1.0, -1, PETSC_FALSE, PETSC_FALSE, &nCells, &cells) >> ablate::utilities::PetscUtilities::checkError;
//        for (PetscInt i = 0; i < nCells; ++i) {
//          PetscScalar *neighborMaskVal = nullptr;
//          xDMPlexPointLocalRef(auxDM, cells[i], vofID, workArray, &neighborMaskVal) >> ablate::utilities::PetscUtilities::checkError;
//          if ( *neighborMaskVal < 0.5 ) {
//            *neighborMaskVal = l + 1;

//            cellMask[reverseCellRange.GetIndex(cells[i])] = l + 1;

//            PetscScalar *vofVal;
//            xDMPlexPointLocalRead(solDM, cells[i], vofID, solArray, &vofVal) >> ablate::utilities::PetscUtilities::checkError;

//            PetscInt nv, *verts;
//            DMPlexCellGetVertices(auxDM, cells[i], &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;

//            for (PetscInt v = 0; v < nv; ++v) {
//              PetscInt id = reverseVertRange.GetIndex(verts[v]);

//              if (vertMask[id]==0) {
//                vertMask[id] = l + 1;

//                PetscScalar *lsVal;
//                xDMPlexPointLocalRef(auxDM, verts[v], lsID, auxArray, &lsVal) >> ablate::utilities::PetscUtilities::checkError;

//                *lsVal = lsRange[ *vofVal < 0.5 ? 1 : 0 ];

//              }
//            }

//            DMPlexCellRestoreVertices(auxDM, cells[i], &nv, &verts);
//          }
//        }
//        DMPlexRestoreNeighbors(solDM, cell, 1, -1.0, -1, PETSC_FALSE, PETSC_FALSE, &nCells, &cells) >> ablate::utilities::PetscUtilities::checkError;
//      }
//    }


//    DMLocalToGlobal(auxDM, workVec, INSERT_VALUES, workVecGlobal) >> utilities::PetscUtilities::checkError;
//    DMGlobalToLocal(auxDM, workVecGlobal, INSERT_VALUES, workVec) >> utilities::PetscUtilities::checkError;
//  }

//  VecRestoreArrayRead(solVec, &solArray) >> ablate::utilities::PetscUtilities::checkError;

//  subDomain->UpdateAuxLocalVector();


//#ifdef saveData
//{
//  sprintf(fname, "mask2_%03ld.txt", saveIter);
//  FILE *f1 = fopen(fname, "w");
//  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
//    const PetscInt cell = cellRange.GetPoint(c);
//    PetscReal x[dim];
//    DMPlexComputeCellGeometryFVM(solDM, cell, NULL, x, NULL) >> ablate::utilities::PetscUtilities::checkError;
//    for (PetscInt d = 0; d < dim; ++d) fprintf(f1, "%+f\t", x[d]);
//    fprintf(f1, "%ld\n", cellMask[c]);
//  }
//  fclose(f1);
//  sprintf(fname, "ls2_%03ld.txt", saveIter);
//  SaveVertexData(auxDM, auxVec, fname, lsField, 1, subDomain);
//}
//#endif



////exit(0);
////PetscPrintf(PETSC_COMM_WORLD, "Reinit\n");
///**************** Level-set reinitialization equation *************************************/
//  const PetscInt vertexNormalID = vertexNormalField->id;
//  const PetscInt curvID = curvField->id;



//  maxDiff = 1.0;
//  iter = 0;

////  while (maxDiff>1.e-2 && iter<(nLevels*5)) {
//  while (maxDiff>1.e-2 && iter<3*(nLevels+1)) {
//    ++iter;

//    for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {
//      if (vertMask[v] > 0) {
//        PetscInt vert = vertRange.GetPoint(v);
//        const PetscReal *phi = nullptr;
//        xDMPlexPointLocalRead(auxDM, vert, lsID, auxArray, &phi) >> ablate::utilities::PetscUtilities::checkError;
//        tempLS[v] = *phi;
//      }
//    }


//    // Determine the current gradient at cells that need updating
//    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
//      if (cellMask[c] > 0) {
//        PetscInt cell = cellRange.GetPoint(c);
//        PetscReal *g = nullptr;
//        xDMPlexPointLocalRef(auxDM, cell, cellNormalID, auxArray, &g) >> ablate::utilities::PetscUtilities::checkError;
//        DMPlexCellGradFromVertex(auxDM, cell, auxVec, lsID, 0, g) >> ablate::utilities::PetscUtilities::checkError;
//      }
//    }

//    for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {
//      if (vertMask[v] > 0) {
//        PetscInt vert = vertRange.GetPoint(v);

//        PetscReal *g = nullptr;
//        xDMPlexPointLocalRef(auxDM, vert, vertexNormalID, auxArray, &g) >> ablate::utilities::PetscUtilities::checkError;
//        DMPlexVertexGradFromVertex(auxDM, vert, auxVec, lsID, 0, g) >> ablate::utilities::PetscUtilities::checkError;

////        // Check if the gradient is zero. This occurs along the skeleton of a shape
////        PetscReal nrm = ablate::utilities::MathUtilities::MagVector(dim, g);
////        if (nrm < PETSC_SMALL) {
////          DMPlexComputeCellGeometryFVM(auxDM, vert, NULL, g, NULL) >> ablate::utilities::PetscUtilities::checkError;
////          for (PetscInt d=0; d<dim; ++d) g[d] += 10.0*PETSC_SMALL;
////        }

//      }
//    }

//    subDomain->UpdateAuxLocalVector();


//    maxDiff = -PETSC_MAX_REAL;

//    for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {

//      if (vertMask[v] > 1) {

//        PetscInt vert = vertRange.GetPoint(v);

//        PetscReal *phi = nullptr;
//        xDMPlexPointLocalRef(auxDM, vert, lsID, auxArray, &phi) >> ablate::utilities::PetscUtilities::checkError;

//        const PetscReal *arrayG = nullptr;
//        xDMPlexPointLocalRef(auxDM, vert, vertexNormalID, auxArray, &arrayG) >> ablate::utilities::PetscUtilities::checkError;

//        PetscReal g[dim];
//        for (PetscInt d = 0; d < dim; ++d){
//          g[d] = arrayG[d];
//        }

//        PetscReal sgn = (*phi)/PetscSqrtReal(PetscSqr(*phi) + PetscSqr(h));

//        if (ablate::utilities::MathUtilities::MagVector(dim, g) < 1.e-10) {
//          *phi = tempLS[v] + 0.5*h*sgn;
//        }
//        else {

//          VertexUpwindGrad(auxDM, auxArray, cellNormalID, vert, PetscSignReal(*phi), g);

//          PetscReal nrm = ablate::utilities::MathUtilities::MagVector(dim, g);

//          *phi = tempLS[v] - 0.5*h*sgn*(nrm - 1.0);

//          // In parallel runs VertexUpwindGrad may return g=0 as there aren't any upwind nodes. Don't incldue that in the diff check
//          if (ablate::utilities::MathUtilities::MagVector(dim, g) > PETSC_SMALL) maxDiff = PetscMax(maxDiff, PetscAbsReal(nrm - 1.0));
//        }


//      }
//    }

//    subDomain->UpdateAuxLocalVector();


//     // Get the maximum change across all processors. This also acts as a sync point
//    MPI_Allreduce(MPI_IN_PLACE, &maxDiff, 1, MPIU_REAL, MPIU_MAX, auxCOMM);


//#ifdef saveData
//    PetscPrintf(PETSC_COMM_WORLD, "Reinit %3" PetscInt_FMT": %e\n", iter, maxDiff);
//#endif

//  }

//#ifdef saveData
//  sprintf(fname, "ls3_%03ld.txt", saveIter);
//  SaveVertexData(auxDM, auxVec, fname, lsField, 1, subDomain);
//#endif

//  // Calculate unit normal vector based on the updated level set values at the vertices
//  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
//    if (cellMask[c] > 0) {
//      PetscInt cell = cellRange.GetPoint(c);
//      PetscScalar *n = nullptr;
//      xDMPlexPointLocalRef(auxDM, cell, cellNormalID, auxArray, &n);
//      DMPlexCellGradFromVertex(auxDM, cell, auxVec, lsID, 0, n) >> ablate::utilities::PetscUtilities::checkError;
//      ablate::utilities::MathUtilities::NormVector(dim, n);
//    }
//  }

//#ifdef saveData
//  sprintf(fname, "mask3_%03ld.txt", saveIter);
//  SaveCellData(auxDM, workVec, fname, vofField, 1, subDomain);
//#endif

//  for (PetscInt c = cellRangeWithoutGhost.start; c < cellRangeWithoutGhost.end; ++c) {
//    PetscInt cell = cellRangeWithoutGhost.GetPoint(c);
//    PetscScalar *H = nullptr;
//    xDMPlexPointLocalRef(auxDM, cell, curvID, auxArray, &H);

//    PetscScalar *maskVal;
//    xDMPlexPointLocalRef(auxDM, cell, vofID, workArray, &maskVal) >> ablate::utilities::PetscUtilities::checkError;

////    if ((PetscAbsScalar(*maskVal - 1.0) < PETSC_SMALL) && ablate::levelSet::Utilities::ValidCell(auxDM, cell)) {
//    if ( (*maskVal > 0.5) && (*maskVal < (nLevels-1)) && ablate::levelSet::Utilities::ValidCell(auxDM, cell)) {
//      CurvatureViaGaussian(auxDM, c - cellRangeWithoutGhost.start, cell, auxVec, lsField, H);
//    }
//    else {
//      *H = 0.0;
//    }
//  }

//  subDomain->UpdateAuxLocalVector();
//#ifdef saveData
//  sprintf(fname, "curv0_%03ld.txt", saveIter);
//  SaveCellData(auxDM, auxVec, fname, curvID, 1, subDomain);
//#endif


//  // Extension
//  PetscInt vertexCurvID = lsID; // Store the vertex curvatures in the work vec at the same location as the level-set


//  for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {
//    if (vertMask[v] > 0 && vertMask[v] < nLevels - 1) {
//      PetscInt vert = vertRange.GetPoint(v);
//      PetscReal *H = nullptr;
//      xDMPlexPointLocalRef(auxDM, vert, vertexCurvID, workArray, &H) >> ablate::utilities::PetscUtilities::checkError;

//      *H = 0.0;

//      PetscInt nCells, *cells, nAve = 0;
//      DMPlexVertexGetCells(auxDM, vert, &nCells, &cells);

//      for (PetscInt c = 0; c < nCells; ++c) {

//        const PetscInt cm = cellMask[reverseCellRange.GetIndex(cells[c])];

//        if (cm > 0 ) {

//          PetscScalar *cellH = nullptr;
//          xDMPlexPointLocalRef(auxDM, cells[c], curvID, auxArray, &cellH);
//          *H += *cellH;
//          ++nAve;
//        }
//      }


//      *H /= nAve;

//      DMPlexVertexRestoreCells(auxDM, vert, &nCells, &cells);

//    }
//  }


//  DMLocalToGlobal(auxDM, workVec, INSERT_VALUES, workVecGlobal) >> utilities::PetscUtilities::checkError;
//  DMGlobalToLocal(auxDM, workVecGlobal, INSERT_VALUES, workVec) >> utilities::PetscUtilities::checkError;



//#ifdef saveData
//  sprintf(fname, "vertH0_%03ld.txt", saveIter);
//  SaveVertexData(auxDM, workVec, fname, lsField, 1, subDomain);
//#endif


//  for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {
//    if (vertMask[v] > 0) {
//      PetscInt vert = vertRange.GetPoint(v);

//      PetscReal *n = nullptr;
//      xDMPlexPointLocalRef(auxDM, vert, vertexNormalID, auxArray, &n) >> ablate::utilities::PetscUtilities::checkError;
//      DMPlexVertexGradFromVertex(auxDM, vert, auxVec, lsID, 0, n) >> ablate::utilities::PetscUtilities::checkError;
//      ablate::utilities::MathUtilities::NormVector(dim, n, n);
//    }
//  }
//  subDomain->UpdateAuxLocalVector();

//  maxDiff = PETSC_MAX_REAL;
//  iter = 0;
//  while ( maxDiff>5e-2 && iter<3*(nLevels+1)) {
//    ++iter;

//    // Curvature gradient at the cell-center
//    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
//      if (cellMask[c] > 0) {
//        PetscInt cell = cellRange.GetPoint(c);
//        PetscScalar *g = nullptr;
//        xDMPlexPointLocalRef(auxDM, cell, cellNormalID, workArray, &g) >> ablate::utilities::PetscUtilities::checkError;
//        DMPlexCellGradFromVertex(auxDM, cell, workVec, vertexCurvID, 0, g);
//      }
//    }

//    maxDiff = -PETSC_MAX_REAL;

//    for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {
//      if (vertMask[v] > 1) {
//        PetscInt vert = vertRange.GetPoint(v);
//        PetscReal g[dim];
//        const PetscReal *phi = nullptr, *n = nullptr;
//        xDMPlexPointLocalRead(auxDM, vert, lsID, auxArray, &phi) >> ablate::utilities::PetscUtilities::checkError;
//        xDMPlexPointLocalRead(auxDM, vert, vertexNormalID, auxArray, &n) >> ablate::utilities::PetscUtilities::checkError;

//        for (PetscInt d = 0; d < dim; ++d) g[d] = n[d];

//        VertexUpwindGrad(auxDM, workArray, cellNormalID, vert, PetscSignReal(*phi), g);

//        PetscReal dH = 0.0;
//        for (PetscInt d = 0; d < dim; ++d) dH += g[d]*n[d];


//        PetscReal *H = nullptr;
//        xDMPlexPointLocalRef(auxDM, vert, vertexCurvID, workArray, &H);

//        PetscReal s = *phi/PetscSqrtReal(PetscSqr(*phi) + h*h);

//        *H -= 0.5*h*s*dH;

//        PetscReal *mag = nullptr;
//        xDMPlexPointLocalRef(auxDM, vert, vertexNormalID, workArray, &mag) >> ablate::utilities::PetscUtilities::checkError;
//        mag[0] = PetscAbsReal(dH);
//      }
//    }

//    DMLocalToGlobal(auxDM, workVec, INSERT_VALUES, workVecGlobal) >> utilities::PetscUtilities::checkError;
//    DMGlobalToLocal(auxDM, workVecGlobal, INSERT_VALUES, workVec) >> utilities::PetscUtilities::checkError;

////     This is temporary until after the review.
////     The norm magnitude is incorrect at the edge of processor domains. There needs to be a way to identify
////      cell which are ghost cells as they will have incorrect answers.

//    for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {
//      if (vertMask[v] > 1) {
//        PetscInt vert = vertRange.GetPoint(v);
//        const PetscReal *mag = nullptr;
//        xDMPlexPointLocalRead(auxDM, vert, vertexNormalID, workArray, &mag) >> ablate::utilities::PetscUtilities::checkError;
//        maxDiff = PetscMax(maxDiff, PetscAbsReal(mag[0]));
//      }
//    }

//     // Get the maximum change across all processors. This also acts as a sync point
//    MPI_Allreduce(MPI_IN_PLACE, &maxDiff, 1, MPIU_REAL, MPIU_MAX, auxCOMM);

//#ifdef saveData
//    PetscPrintf(PETSC_COMM_WORLD, "Extension %3" PetscInt_FMT": %e\n", iter, maxDiff);
//#endif
//  }


//#ifdef saveData
//  sprintf(fname, "vertH1_%03ld.txt", saveIter);
//  SaveVertexData(auxDM, workVec, fname, lsField, 1, subDomain);
//#endif



//   for (PetscInt iter = 0; iter < 5; ++iter) {

//    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
//      if (cellMask[c] > 0) {
//        PetscInt cell = cellRange.GetPoint(c);
//        PetscScalar *g = nullptr;
//        xDMPlexPointLocalRef(auxDM, cell, cellNormalID, workArray, &g) >> ablate::utilities::PetscUtilities::checkError;
//        DMPlexCellGradFromVertex(auxDM, cell, workVec, vertexCurvID, 0, g);

//        const PetscScalar *n = nullptr;
//        xDMPlexPointLocalRead(auxDM, cell, cellNormalID, auxArray, &n);

//        const PetscReal dot = ablate::utilities::MathUtilities::DotVector(dim, n, g);

//        for (PetscInt d = 0; d < dim; ++d) g[d] -= dot*n[d];

//      }
//    }
//    DMLocalToGlobal(auxDM, workVec, INSERT_VALUES, workVecGlobal) >> utilities::PetscUtilities::checkError;
//    DMGlobalToLocal(auxDM, workVecGlobal, INSERT_VALUES, workVec) >> utilities::PetscUtilities::checkError;


//    for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {
//      if (vertMask[v] > 0) {
//        PetscInt vert = vertRange.GetPoint(v);
//        PetscReal div = 0.0;

//        for (PetscInt d = 0; d < dim; ++d) {
//          PetscReal g[dim];
//          DMPlexVertexGradFromCell(auxDM, vert, workVec, cellNormalID, d, g);
//          div += g[d];
//        }

//        PetscReal *H = nullptr;
//        xDMPlexPointLocalRef(auxDM, vert, vertexCurvID, workArray, &H);

//        *H += 0.5*h*h*div;

//      }
//    }
//    DMLocalToGlobal(auxDM, workVec, INSERT_VALUES, workVecGlobal) >> utilities::PetscUtilities::checkError;
//    DMGlobalToLocal(auxDM, workVecGlobal, INSERT_VALUES, workVec) >> utilities::PetscUtilities::checkError;
//  }



//#ifdef saveData
//  sprintf(fname, "vertH2_%03ld.txt", saveIter);
//  SaveVertexData(auxDM, workVec, fname, lsField, 1, subDomain);
//#endif



//  // Now set the curvature at the cell-center via averaging

//  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
//    if (cellMask[c] > 0) {
//      PetscInt cell = cellRange.GetPoint(c);

//      PetscScalar *cellH = nullptr;
//      xDMPlexPointLocalRef(auxDM, cell, curvID, auxArray, &cellH) >> utilities::PetscUtilities::checkError;

//      *cellH = 0.0;

//      PetscInt nv, *verts;
//      DMPlexCellGetVertices(auxDM, cell, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;
//      for (PetscInt i = 0; i < nv; ++i) {
//        const PetscReal *H;
//        xDMPlexPointLocalRead(auxDM, verts[i], vertexCurvID, workArray, &H) >> utilities::PetscUtilities::checkError;
//        *cellH += *H;
//      }
//      *cellH /= nv;

//      DMPlexCellRestoreVertices(auxDM, cell, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;
//    }
//  }


//  subDomain->UpdateAuxLocalVector();

//#ifdef saveData
//  sprintf(fname, "cellH0_%03ld.txt", saveIter);
//  SaveVertexData(auxDM, workVec, fname, lsField, 1, subDomain);
//  sprintf(fname, "cellNormal1_%03ld.txt", saveIter);
//  SaveCellData(auxDM, auxVec, fname, cellNormalField, dim, subDomain);

//#endif

//  VecRestoreArray(workVec, &workArray);
//  DMRestoreLocalVector(auxDM, &workVec) >> utilities::PetscUtilities::checkError;
//  DMRestoreGlobalVector(auxDM, &workVecGlobal) >> utilities::PetscUtilities::checkError;



//  // Cleanup all memory
//  tempLS += vertRange.start;
//  DMRestoreWorkArray(auxDM, vertRange.end - vertRange.start, MPIU_SCALAR, &tempLS) >> ablate::utilities::PetscUtilities::checkError;
//  vertMask += vertRange.start; // Reset the offset, otherwise DMRestoreWorkArray will return unexpected results
//  DMRestoreWorkArray(auxDM, vertRange.end - vertRange.start, MPIU_INT, &vertMask) >> ablate::utilities::PetscUtilities::checkError;
//  cellMask += cellRange.start; // Reset the offset, otherwise DMRestoreWorkArray will return unexpected results
//  DMRestoreWorkArray(solDM, cellRange.end - cellRange.start, MPIU_INT, &cellMask) >> ablate::utilities::PetscUtilities::checkError;

//  subDomain->RestoreRange(vertRange);
//  subDomain->RestoreRange(cellRange);
//  flow.RestoreRange(cellRangeWithoutGhost);

//  VecRestoreArray(auxVec, &auxArray) >> utilities::PetscUtilities::checkError;

//}




  if (subpointIndices) ISRestoreIndices(subpointIS, &subpointIndices) >> utilities::PetscUtilities::checkError;

}


