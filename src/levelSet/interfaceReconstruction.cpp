#include "interfaceReconstruction.hpp"
#include <petsc.h>
#include "utilities/petscSupport.hpp"
#include "utilities/mathUtilities.hpp"
#include "domain/fieldAccessor.hpp"
#include "levelSetUtilities.hpp"
#include "utilities/constants.hpp"



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

  nGaussStencil = PetscPowInt(gaussianNQuad, dim); // The number of cells in the integration stencil

  const PetscInt nGaussRange[3] = {gaussianNQuad, (dim > 1) ? gaussianNQuad : 1, (dim > 2) ? gaussianNQuad : 1};

  PetscMalloc1(nGaussStencil*nTotalCell, &interpCellList) >> ablate::utilities::PetscUtilities::checkError;

  for (PetscInt c = 0; c < nTotalCell; ++c) {

    const PetscInt cell = cellList[c];

    PetscReal x0[3] = {0.0, 0.0, 0.0};
    DMPlexComputeCellGeometryFVM(cellDM, cell, NULL, x0, NULL) >> ablate::utilities::PetscUtilities::checkError;

    PetscInt nCells, *cellList;
    DMPlexGetNeighbors(cellDM, cell, 2, -1.0, -1, PETSC_FALSE, PETSC_FALSE, &nCells, &cellList) >> ablate::utilities::PetscUtilities::checkError;

    for (PetscInt i = 0; i < nGaussRange[0]; ++i) {
      for (PetscInt j = 0; j < nGaussRange[1]; ++j) {
        for (PetscInt k = 0; k < nGaussRange[2]; ++k) {

          PetscReal x[3] = {x0[0] + sigma*gaussianQuad[i], x0[1] + sigma*gaussianQuad[j], x0[2] + sigma*gaussianQuad[k]};

          const PetscInt interpCell = FindCell(cellDM, dim, x, nCells, cellList, NULL);

          if (interpCell < 0) {
            int rank;
            MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
            throw std::runtime_error("BuildInterpCellList could not determine the location of (" + std::to_string(x[0]) + ", " + std::to_string(x[1]) + ", " + std::to_string(x[2]) + ") on rank " + std::to_string(rank) + ".");
          }

          interpCellList[c*nGaussStencil + gaussianNQuad*(k*gaussianNQuad + j) + i] = interpCell;
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
//
//  The numbering is the following:
//    1: Cut-cells or vertices associated with cut-cells
//    2 -> nLevels: Neighbors of cut-cells in increasing distance from interface
//   -1: The cells/vertices directly next to those labelled as nLevels.
//    0: Cells/vertices far from the interface
void Reconstruction::SetMasks(const PetscInt nLevels, PetscInt *cellMask, PetscInt *vertMask, Vec vofVec[2]) {

  PetscScalar *vofArray = nullptr, *cellMaskVecArray = nullptr;
  Vec cellMaskVec[2] = {nullptr, nullptr};



  DMGetLocalVector(cellDM, &cellMaskVec[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
  DMGetGlobalVector(cellDM, &cellMaskVec[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;

  VecGetArray(vofVec[LOCAL], &vofArray) >> ablate::utilities::PetscUtilities::checkError;
  VecGetArray(cellMaskVec[LOCAL], &cellMaskVecArray);
  for (PetscInt c = 0; c < nTotalCell; ++c) {
    cellMaskVecArray[c] = cellMask[c] = ((vofArray[c] > 0.001) && (vofArray[c] < 0.999));
  }


//   Turn off any "cut cells" where the cell is not surrounded by any other cut cells.
//   To avoid cut-cells two cells-thick turn off any cut-cells which have a neighoring gradient passing through them.
  const PetscInt    dim = subDomain->GetDimensions();
  for (PetscInt c = 0; c < nTotalCell; ++c) {

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

      cellMask[c] = (nCut>1); // If nCut equals 1 then the center cell is the only cut cell, so deactivate it

      PetscScalar n[dim];
      DMPlexCellGradFromCell(cellDM, cell, vofVec[LOCAL], -1, 0, n) >> ablate::utilities::PetscUtilities::checkError;

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
  VecRestoreArray(cellMaskVec[LOCAL], &cellMaskVecArray) >> ablate::utilities::PetscUtilities::checkError;


  // Now label the surrounding cells
  for (PetscInt l = 1; l <= nLevels; ++l) {

    VecZeroEntries(cellMaskVec[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
    VecGetArray(cellMaskVec[LOCAL], &cellMaskVecArray) >> ablate::utilities::PetscUtilities::checkError;
    for (PetscInt c = 0; c < nLocalCell; ++c) {
      if ( cellMask[c] == l ) {
        PetscInt cell = cellList[c];
        PetscInt nCells, *cells;
        DMPlexGetNeighbors(cellDM, cell, 1, -1.0, -1, PETSC_FALSE, PETSC_FALSE, &nCells, &cells) >> ablate::utilities::PetscUtilities::checkError;
        for (PetscInt i = 0; i < nCells; ++i) {
          const PetscInt id = reverseCellList[cells[i]];
          ++cellMaskVecArray[id];
        }
        DMPlexRestoreNeighbors(cellDM, cell, 1, -1.0, -1, PETSC_FALSE, PETSC_FALSE, &nCells, &cells) >> ablate::utilities::PetscUtilities::checkError;
      }
    }
    VecRestoreArray(cellMaskVec[LOCAL], &cellMaskVecArray) >> ablate::utilities::PetscUtilities::checkError;

    VecZeroEntries(cellMaskVec[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;
    DMLocalToGlobal(cellDM, cellMaskVec[LOCAL], ADD_VALUES, cellMaskVec[GLOBAL]) >> utilities::PetscUtilities::checkError;
    DMGlobalToLocal(cellDM, cellMaskVec[GLOBAL], INSERT_VALUES, cellMaskVec[LOCAL]) >> utilities::PetscUtilities::checkError;

    const PetscInt setValue = (l == nLevels) ? -1 : l + 1;

    VecGetArray(cellMaskVec[LOCAL], &cellMaskVecArray) >> ablate::utilities::PetscUtilities::checkError;
    for (PetscInt c = 0; c < nTotalCell; ++c) {
      if (cellMask[c] == 0 && cellMaskVecArray[c]>0.5) cellMask[c] = setValue;
    }
    VecRestoreArray(cellMaskVec[LOCAL], &cellMaskVecArray) >> ablate::utilities::PetscUtilities::checkError;
  }

  // Set the vertex mask
  for (PetscInt v = 0; v < nTotalVert; ++v) vertMask[v] = nLevels + 2;

  // First set the vertices associated with marked cells
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

  // Next set the additional vertices associated with boundary cells
  for (PetscInt c = 0; c < nTotalCell; ++c) {
    if (cellMask[c] == -1) {
      const PetscInt cell = cellList[c];

      PetscInt nVert, *verts;
      DMPlexCellGetVertices(vertDM, cell, &nVert, &verts) >> ablate::utilities::PetscUtilities::checkError;

      for (PetscInt v = 0; v < nVert; ++v) {
        const PetscInt id = reverseVertList[verts[v]];
        vertMask[id] = (vertMask[id]==(nLevels+2)) ? -1 : vertMask[id];
      }
      DMPlexCellRestoreVertices(vertDM, cell, &nVert, &verts) >> ablate::utilities::PetscUtilities::checkError;
    }
  }

  // Switch all deactivated vertices to 0
  for (PetscInt v = 0; v < nTotalVert; ++v) {
    vertMask[v] = (vertMask[v]==(nLevels+2)) ? 0 : vertMask[v];
  }

  DMRestoreLocalVector(cellDM, &cellMaskVec[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
  DMRestoreGlobalVector(cellDM, &cellMaskVec[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;

}

// vofVec MUST have ghost cell information
void Reconstruction::InitalizeLevelSet(Vec vofVec, const PetscInt *cellMask, const PetscInt *vertMask, Vec lsVec[2]) {

  MPI_Comm lsCOMM = PetscObjectComm((PetscObject)vertDM);

  const PetscInt dim = subDomain->GetDimensions();

  // First get the number of cut-cells associated with each vertex
  PetscInt *lsCount = nullptr;
  DMGetWorkArray(vertDM, nLocalVert, MPIU_INT, &lsCount) >> ablate::utilities::PetscUtilities::checkError;
  for (PetscInt v = 0; v < nLocalVert; ++v) {

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
  PetscScalar *cellGrad = nullptr;

  DMGetWorkArray(cellGradDM, nTotalCell*dim, MPIU_REAL, &cellGrad) >> ablate::utilities::PetscUtilities::checkError;
  for (PetscInt c = 0; c < nTotalCell; ++c) {
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

    for (PetscInt c = 0; c < nTotalCell; ++c) {

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

        for (PetscInt v = 0; v < nv; ++v) {
          const PetscInt id = reverseVertList[verts[v]];
          if (id < nLocalVert){
            lsArray[GLOBAL][id] += lsVertVals[v];
          }
        }

        DMRestoreWorkArray(vertDM, nv, MPIU_REAL, &lsVertVals) >> ablate::utilities::PetscUtilities::checkError;
        DMPlexCellRestoreVertices(vertDM, cell, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;
      }
    }

    VecGetArray(lsVec[LOCAL], &lsArray[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;

    maxDiff = -1.0;
    for (PetscInt v = 0; v < nLocalVert; ++v) {
      if (vertMask[v] == 1) {
        lsArray[GLOBAL][v] /= lsCount[v];
        maxDiff = PetscMax(maxDiff, PetscAbsReal(lsArray[GLOBAL][v] - lsArray[LOCAL][v]));
      }
    }

    // Get the maximum change across all processors. This also acts as a sync point
    MPI_Allreduce(MPI_IN_PLACE, &maxDiff, 1, MPIU_REAL, MPIU_MAX, lsCOMM);

    VecRestoreArray(lsVec[GLOBAL], &lsArray[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;
    VecRestoreArray(lsVec[LOCAL], &lsArray[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;

    // It is necessary to communicate the updates, otherwise errors at the edge of ghost cells will
    //    propogate through the domain.
    DMGlobalToLocal(vertDM, lsVec[GLOBAL], INSERT_VALUES, lsVec[LOCAL]) >> utilities::PetscUtilities::checkError;

    // Update the cell-center normal using the level-set data
    for (PetscInt c = 0; c < nTotalCell; ++c) {
      if (cellMask[c] == 1) {
        DMPlexCellGradFromVertex(vertDM, cellList[c], lsVec[LOCAL], -1, 0, &cellGrad[c*dim]) >> ablate::utilities::PetscUtilities::checkError;
        ablate::utilities::MathUtilities::NormVector(dim, &cellGrad[c*dim]);
      }
    }

//    PetscPrintf(PETSC_COMM_WORLD, "Cut Cells %" PetscInt_FMT": %+e\n", iter, maxDiff) >> ablate::utilities::PetscUtilities::checkError;

  }


  DMRestoreWorkArray(vertDM, nLocalVert, MPIU_INT, &lsCount) >> ablate::utilities::PetscUtilities::checkError;
  DMRestoreWorkArray(cellGradDM, nTotalCell*dim, MPIU_REAL, &cellGrad) >> ablate::utilities::PetscUtilities::checkError;

  if (maxDiff > 1e-3*h) {
    throw std::runtime_error("Interface reconstruction has failed.\n");
  }


  /*********   Set the values in the rest of the domain ******************/

  // Range of level-set values
  PetscReal lsRange[2] = {-PETSC_MAX_REAL, PETSC_MAX_REAL};
  VecMin(lsVec[GLOBAL], NULL, &lsRange[0]);
  VecMax(lsVec[GLOBAL], NULL, &lsRange[1]);

  // Maximum distance in the domain
  PetscReal gMin[3], gMax[3];
  DMGetBoundingBox(vertDM, gMin, gMax) >> ablate::utilities::PetscUtilities::checkError;

  PetscReal maxDist = 0.0;
  for (PetscInt d = 0; d < dim; ++d) {
    maxDist += PetscSqr(gMax[d] - gMin[d]);
  }
  maxDist = PetscSqrtReal(maxDist);

  maxDist = PETSC_MAX_REAL;


  PetscScalar *lsGlobalArray = nullptr;
  VecGetArray(lsVec[GLOBAL], &lsGlobalArray) >> ablate::utilities::PetscUtilities::checkError;
  for (PetscInt c = 0 ; c < nLocalCell; ++c) {
    const PetscInt cell = cellList[c];
    PetscInt nVerts, *verts;

    DMPlexCellGetVertices(vertDM, cell, &nVerts, &verts) >> utilities::PetscUtilities::checkError;

    const PetscReal lsSetValues[2] = {lsRange[ vofArray[c] < 0.5 ? 1 : 0 ], PetscSignReal(0.5 - vofArray[c])*maxDist};

    for (PetscInt v = 0; v < nVerts; ++v) {
      const PetscInt id = reverseVertList[verts[v]];

      if (id < nLocalVert) {

        if(vertMask[id] > 1) lsGlobalArray[id] = lsSetValues[0];
        else if (vertMask[id] <= 0) lsGlobalArray[id] = lsSetValues[1];
      }
    }

    DMPlexCellRestoreVertices(vertDM, cell, &nVerts, &verts) >> utilities::PetscUtilities::checkError;
  }

  VecRestoreArrayRead(vofVec, &vofArray) >> ablate::utilities::PetscUtilities::checkError;
  VecRestoreArray(lsVec[GLOBAL], &lsGlobalArray) >> ablate::utilities::PetscUtilities::checkError;

  DMGlobalToLocal(vertDM, lsVec[GLOBAL], INSERT_VALUES, lsVec[LOCAL]) >> utilities::PetscUtilities::checkError;




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



/**
  * Compute the upwind derivative at a vertex
  * @param dm - Domain of the gradient data.
  * @param gradArray - Array containing the cell-centered gradient
  * @param v - Vertex id
  * @param direction - The direction to be considered upwind. +1 for standard upwind, -1 of downwind
  * @param g - On input the gradient of the level-set field at a vertex. On output the upwind gradient at v
  */
void Reconstruction::VertexUpwind(const PetscScalar *gradArray, const PetscInt v, const PetscReal direction, const PetscInt *cellMask, PetscReal *g) {
  // The upwind direction is determined using the dot product between the vector u and the vector connecting the cell-center
  //    and the vertex

  const PetscInt    dim = subDomain->GetDimensions();
  PetscReal         weightTotal = 0.0;
  PetscScalar       x0[3] = {0.0, 0.0, 0.0}, n[3] = {0.0, 0.0, 0.0};

  ablate::utilities::MathUtilities::NormVector(dim, g, n);

  DMPlexComputeCellGeometryFVM(vertDM, v, NULL, x0, NULL) >> ablate::utilities::PetscUtilities::checkError;

  for (PetscInt d = 0; d < dim; ++d) {
    g[d] = 0.0;
  }

  // Obtain all cells which use this vertex
  PetscInt nCells, *cells;
  DMPlexVertexGetCells(vertDM, v, &nCells, &cells) >> ablate::utilities::PetscUtilities::checkError;

  for (PetscInt c = 0; c < nCells; ++c) {

    const PetscInt id = reverseCellList[cells[c]];

    if (cellMask[id] > 0) {

      PetscReal x[3];
      DMPlexComputeCellGeometryFVM(vertDM, cells[c], NULL, x, NULL) >> ablate::utilities::PetscUtilities::checkError;

      ablate::utilities::MathUtilities::Subtract(dim, x0, x, x);
      ablate::utilities::MathUtilities::NormVector(dim, x, x);
      PetscReal dot = ablate::utilities::MathUtilities::DotVector(dim, n, x);

      dot *= direction;

      if (dot>0.0) {

        weightTotal += dot;

        // Weighted average of the surrounding cell-center gradients.
        //  Note that technically this is (in 2D) the area of the quadrilateral that is formed by connecting
        //  the vertex, center of the neighboring edges, and the center of the triangle. As the three quadrilaterals
        //  that are formed this way all have the same area, there is no need to take into account the 1/3. Something
        //  similar should hold in 3D and for other cell types that ABLATE uses.
        for (PetscInt d = 0; d < dim; ++d) {
          g[d] += dot*gradArray[id*dim + d];
        }
      }
    }
  }

  DMPlexVertexRestoreCells(vertDM, v, &nCells, &cells) >> ablate::utilities::PetscUtilities::checkError;

  // Size of the communicator
//  MPI_Comm comm = PetscObjectComm((PetscObject)dm);
//  int size;
//  MPI_Comm_size(comm, &size) >> ablate::utilities::MpiUtilities::checkError;

  // Error checking
  if ( PetscAbs(weightTotal) < ablate::utilities::Constants::small ) {
    // When running on a single processor all vertices should have an upwind cell. Throw an error if that's not the case.
    // When running in parallel, ghost vertices at the edge of the local domain may not have any surrounding upwind cells, so
    //  ignore the error and simply set the upwind gradient to zero.
//    if ( size==1 ) {
//      throw std::runtime_error("ablate::levelSet::Utilities::VertexUpwindGrad encounted a situation where there are no upwind cells");
//    }
//    if ( size==1 ) {
//      char err[255];
//      sprintf(err, "ablate::levelSet::Utilities::VertexUpwindGrad encounted a situation where there are no upwind cells %f,%f", x0[0], x0[1]);
//      throw std::runtime_error(err);
//    }
    for (PetscInt d = 0; d < dim; ++d) {
      g[d] = 0.0;
    }
  }
  else {
    for (PetscInt d = 0; d < dim; ++d) {
      g[d] /= weightTotal;
    }
  }
}

void Reconstruction::ReinitializeLevelSet(const PetscInt *cellMask, const PetscInt *vertMask, Vec lsVec[2]) {

  const PetscInt  dim = subDomain->GetDimensions();
  PetscReal       maxDiff = 1.0;
  PetscInt        iter = 0;
  PetscReal       *cellGrad = nullptr, *vertGrad = nullptr;
  PetscReal       h = 0.0;
  MPI_Comm        lsCOMM = PetscObjectComm((PetscObject)vertDM);

  DMPlexGetMinRadius(vertDM, &h) >> ablate::utilities::PetscUtilities::checkError;
  h *= 2.0; // Min radius returns the distance between a cell-center and a face. Double it to get the average cell size


  DMGetWorkArray(cellGradDM, nTotalCell*dim, MPIU_REAL, &cellGrad) >> ablate::utilities::PetscUtilities::checkError;
  DMGetWorkArray(vertGradDM, nTotalVert*dim, MPIU_REAL, &vertGrad) >> ablate::utilities::PetscUtilities::checkError;

//  const PetscInt maxIter = 3*(nLevels+1);
  const PetscInt maxIter = 1000;

  while (maxDiff>1.e-3 && iter<maxIter) {
    ++iter;

    PetscScalar *lsArray = nullptr;

    // Determine the current gradient at cells that need updating
    for (PetscInt c = 0; c < nTotalCell; ++c) {
      if (cellMask[c] > 0) {
        DMPlexCellGradFromVertex(vertDM, cellList[c], lsVec[LOCAL], -1, 0, &cellGrad[c*dim]) >> ablate::utilities::PetscUtilities::checkError;
      }
    }


    // Gradient at vertices
    //  For stability reasons this is written as the average of the surrounding activated cells
    for (PetscInt v = 0; v < nTotalVert; ++v) {
      if (vertMask[v] > 0) {

        for (PetscInt d = 0; d < dim; ++d) vertGrad[v*dim + d] = 0.0;

        PetscInt nc = 0;

        PetscInt nCells, *cells;
        DMPlexVertexGetCells(vertDM, vertList[v], &nCells, &cells);
        for (PetscInt c = 0; c < nCells; ++c) {
          const PetscInt id = reverseCellList[cells[c]];
          if (cellMask[id] > 0 ) {
            for (PetscInt d = 0; d < dim; ++d) vertGrad[v*dim + d] += cellGrad[id*dim + d];
            ++nc;
          }
        }
        DMPlexVertexRestoreCells(vertDM, vertList[v], &nCells, &cells);

        if (nc==0) throw std::runtime_error("Vertex has no valid surrounding cells!\n");

        for (PetscInt d = 0; d < dim; ++d) vertGrad[v*dim + d] /= nc;

//        DMPlexVertexGradFromVertex(vertDM, vertList[v], lsVec[LOCAL], -1, 0, &vertGrad[v*dim]) >> ablate::utilities::PetscUtilities::checkError;
      }
    }

    maxDiff = -PETSC_MAX_REAL;

    VecGetArray(lsVec[GLOBAL], &lsArray);
    for (PetscInt v = 0; v < nLocalVert; ++v) {

      if (vertMask[v] > 1) {
        const PetscInt vert = vertList[v];
        const PetscReal oldPhi = lsArray[v];

        PetscReal *g = &vertGrad[v*dim];

        PetscReal sgn = (oldPhi)/PetscSqrtReal(PetscSqr(oldPhi) + PetscSqr(h));

        if (ablate::utilities::MathUtilities::MagVector(dim, g) < 1.e-10) {
          lsArray[v] += 0.5*h*sgn;
        }
        else {

          VertexUpwind(cellGrad, vert, PetscSignReal(oldPhi), cellMask, g);

          PetscReal nrm = ablate::utilities::MathUtilities::MagVector(dim, g);

          lsArray[v] -=0.5*h*sgn*(nrm - 1.0);

          // In parallel runs VertexUpwind may return g=0 as there aren't any upwind nodes. Don't incldue that in the diff check
          if (ablate::utilities::MathUtilities::MagVector(dim, g) > PETSC_SMALL) maxDiff = PetscMax(maxDiff, PetscAbsReal(nrm - 1.0));
        }
      }
    }
    VecRestoreArray(lsVec[GLOBAL], &lsArray);

    DMGlobalToLocal(vertDM, lsVec[GLOBAL], INSERT_VALUES, lsVec[LOCAL]) >> utilities::PetscUtilities::checkError;

     // Get the maximum change across all processors. This also acts as a sync point
    MPI_Allreduce(MPI_IN_PLACE, &maxDiff, 1, MPIU_REAL, MPIU_MAX, lsCOMM);

    PetscPrintf(PETSC_COMM_WORLD, "Reinit %3" PetscInt_FMT": %e\n", iter, maxDiff);
  }

  DMRestoreWorkArray(cellGradDM, nTotalCell*dim, MPIU_REAL, &cellGrad) >> ablate::utilities::PetscUtilities::checkError;
  DMRestoreWorkArray(vertGradDM, nTotalVert*dim, MPIU_REAL, &vertGrad) >> ablate::utilities::PetscUtilities::checkError;

}

static PetscReal Reconstruction_GaussianDerivativeFactor(const PetscInt dim, const PetscReal *x, const PetscReal s,  const PetscInt dx, const PetscInt dy, const PetscInt dz) {

  const PetscReal s2 = PetscSqr(s);

  const PetscInt derHash = 100*dx + 10*dy + dz;

  if (derHash > 0 && PetscAbsReal(s)<PETSC_SMALL) return (0.0);

  switch (derHash) {
    case   0: // Value
      return (1.0);
    case 100: // x
      return (x[0]/s2);
    case  10: // y
      return (x[1]/s2);
    case   1: // z
      return (x[2]/s2);
    case 200: // xx
      return ((x[0]*x[0] - s2)/PetscSqr(s2));
    case  20: // yy
      return ((x[1]*x[1] - s2)/PetscSqr(s2));
    case   2: // zz
      return ((x[2]*x[2] - s2)/PetscSqr(s2));
    case 110: // xy
      return (x[0]*x[1]/PetscSqr(s2));
    case 101: // xz
      return (x[0]*x[2]/PetscSqr(s2));
    case  11: // yz
      return (x[1]*x[2]/PetscSqr(s2));
    default:
      throw std::runtime_error("Unknown derivative request");
  }

}

static PetscReal CurvatureViaGaussian_1D(DM dm, const PetscInt nGaussStencil, const PetscInt gaussianNQuad, const PetscReal gaussianQuad[], const PetscReal gaussianWeights[], const PetscReal sigma, const PetscInt interpCellList[], std::shared_ptr<ablate::domain::rbf::RBF> vertRBF, const PetscInt c, const PetscInt cell, const Vec lsVec) {
  return 0.0;
}

static PetscReal CurvatureViaGaussian_2D(DM dm, const PetscInt nGaussStencil, const PetscInt gaussianNQuad, const PetscReal gaussianQuad[], const PetscReal gaussianWeights[], const PetscReal sigma, const PetscInt interpCellList[], std::shared_ptr<ablate::domain::rbf::RBF> vertRBF, const PetscInt c, const PetscInt cell, const Vec lsVec) {
  PetscReal cx = 0.0, cy = 0.0, cxx = 0.0, cyy = 0.0, cxy = 0.0;
  const PetscInt dim = 2;

  PetscReal x0[dim];
  DMPlexComputeCellGeometryFVM(dm, cell, NULL, x0, NULL) >> ablate::utilities::PetscUtilities::checkError;

  for (PetscInt i = 0; i < gaussianNQuad; ++i) {
    for (PetscInt j = 0; j < gaussianNQuad; ++j) {

      const PetscReal dist[2] = {sigma*gaussianQuad[i], sigma*gaussianQuad[j]};
      PetscReal x[2] = {x0[0] + dist[0], x0[1] + dist[1]};

      const PetscInt interpCell = interpCellList[c*nGaussStencil + gaussianNQuad*j + i];
      const PetscReal lsVal = vertRBF->Interpolate(dm, -1, lsVec, interpCell, x);

      const PetscReal wt = gaussianWeights[i]*gaussianWeights[j];

      cx  += wt*Reconstruction_GaussianDerivativeFactor(dim, dist, sigma, 1, 0, 0)*lsVal;
      cy  += wt*Reconstruction_GaussianDerivativeFactor(dim, dist, sigma, 0, 1, 0)*lsVal;
      cxx += wt*Reconstruction_GaussianDerivativeFactor(dim, dist, sigma, 2, 0, 0)*lsVal;
      cyy += wt*Reconstruction_GaussianDerivativeFactor(dim, dist, sigma, 0, 2, 0)*lsVal;
      cxy += wt*Reconstruction_GaussianDerivativeFactor(dim, dist, sigma, 1, 1, 0)*lsVal;
    }
  }

  const PetscReal H = (cxx*cy*cy + cyy*cx*cx - 2.0*cxy*cx*cy)/PetscPowReal(cx*cx + cy*cy, 1.5);

  return H;
}

static PetscReal CurvatureViaGaussian_3D(DM dm, const PetscInt nGaussStencil, const PetscInt gaussianNQuad, const PetscReal gaussianQuad[], const PetscReal gaussianWeights[], const PetscReal sigma, const PetscInt interpCellList[], std::shared_ptr<ablate::domain::rbf::RBF> vertRBF, const PetscInt c, const PetscInt cell, const Vec lsVec) {
  PetscReal cx = 0.0, cy = 0.0, cz = 0.0, cxx = 0.0, cyy = 0.0, czz = 0.0, cxy = 0.0, cxz = 0.0, cyz = 0.0;
  const PetscInt dim = 3;

  PetscReal x0[dim];
  DMPlexComputeCellGeometryFVM(dm, cell, NULL, x0, NULL) >> ablate::utilities::PetscUtilities::checkError;

  for (PetscInt i = 0; i < gaussianNQuad; ++i) {
    for (PetscInt j = 0; j < gaussianNQuad; ++j) {
      for (PetscInt k = 0; k < gaussianNQuad; ++k) {

        const PetscReal dist[3] = {sigma*gaussianQuad[i], sigma*gaussianQuad[j], sigma*gaussianQuad[k]};
        PetscReal x[3] = {x0[0] + dist[0], x0[1] + dist[1], x0[2] + dist[2]};

        const PetscInt interpCell = interpCellList[c*nGaussStencil + gaussianNQuad*(k*gaussianNQuad + j) + i];
        const PetscReal lsVal = vertRBF->Interpolate(dm, -1, lsVec, interpCell, x);

        const PetscReal wt = gaussianWeights[i]*gaussianWeights[j]*gaussianWeights[k];

        cx  += wt*Reconstruction_GaussianDerivativeFactor(dim, dist, sigma, 1, 0, 0)*lsVal;
        cy  += wt*Reconstruction_GaussianDerivativeFactor(dim, dist, sigma, 0, 1, 0)*lsVal;
        cz  += wt*Reconstruction_GaussianDerivativeFactor(dim, dist, sigma, 0, 0, 1)*lsVal;
        cxx += wt*Reconstruction_GaussianDerivativeFactor(dim, dist, sigma, 2, 0, 0)*lsVal;
        cyy += wt*Reconstruction_GaussianDerivativeFactor(dim, dist, sigma, 0, 2, 0)*lsVal;
        czz += wt*Reconstruction_GaussianDerivativeFactor(dim, dist, sigma, 0, 0, 2)*lsVal;
        cxy += wt*Reconstruction_GaussianDerivativeFactor(dim, dist, sigma, 1, 1, 0)*lsVal;
        cxz += wt*Reconstruction_GaussianDerivativeFactor(dim, dist, sigma, 1, 0, 1)*lsVal;
        cyz += wt*Reconstruction_GaussianDerivativeFactor(dim, dist, sigma, 0, 1, 1)*lsVal;
      }
    }
  }

  const PetscReal H = (cxx*(cy*cy + cz*cz) + cyy*(cx*cx + cz*cz) + czz*(cx*cx + cy*cy) - 2.0*(cxy*cx*cy + cxz*cx*cz + cyz*cy*cz))/PetscPowReal(cx*cx + cy*cy + cz*cz, 1.5);

  return H;
}

void Reconstruction::CalculateCellCurvatures(const PetscInt *cellMask, const PetscInt *vertMask, Vec lsVec[2], Vec curvVec[2]) {

  PetscReal (*curvFcn)(DM dm, const PetscInt nGaussStencil, const PetscInt gaussianNQuad, const PetscReal gaussianQuad[], const PetscReal gaussianWeights[], const PetscReal sigma, const PetscInt interpCellList[], std::shared_ptr<ablate::domain::rbf::RBF> vertRBF, const PetscInt c, const PetscInt cell, const Vec lsVec) = nullptr;
  const PetscInt  dim = subDomain->GetDimensions();

  switch (dim) {
    case 1:
      curvFcn = &CurvatureViaGaussian_1D;
      break;
    case 2:
      curvFcn = &CurvatureViaGaussian_2D;
      break;
    case 3:
      curvFcn = &CurvatureViaGaussian_3D;
      break;
    default:
      throw std::runtime_error("Reconstruction::CalculateCellCurvatures does not work for domains with dimentions " + std::to_string(dim) + ".");
  }

  PetscReal h;
  DMPlexGetMinRadius(vertDM, &h) >> ablate::utilities::PetscUtilities::checkError;
  h *= 2.0;
  const PetscReal sigma = sigmaFactor*h;

  VecZeroEntries(curvVec[GLOBAL]) >> utilities::PetscUtilities::checkError;
  PetscScalar *curvature;
  VecGetArray(curvVec[GLOBAL], &curvature) >> utilities::PetscUtilities::checkError;
  for (PetscInt c = 0; c < nLocalCell; ++c) {
    if (cellMask[c] > 0 && cellMask[c] < 6) {
      curvature[c] = curvFcn(vertDM, nGaussStencil, gaussianNQuad, gaussianQuad, gaussianWeights, sigma, interpCellList, vertRBF, c, cellList[c], lsVec[LOCAL]);
    }
  }
  VecRestoreArray(curvVec[GLOBAL], &curvature) >> utilities::PetscUtilities::checkError;
  DMGlobalToLocal(cellDM, curvVec[GLOBAL], INSERT_VALUES, curvVec[LOCAL]) >> utilities::PetscUtilities::checkError;
}



/**
  * Compute the upwind derivative at a cell-center
  * @param dm - Domain of the gradient data.
  * @param gradArray - Array containing the vertex-based gradient
  * @param c - Cell id
  * @param direction - The direction to be considered upwind. +1 for standard upwind, -1 of downwind
  * @param g - On input the gradient of the field at the cell-center. On output the upwind gradient at c
  */
void Reconstruction::CellUpwind(const PetscScalar *gradArray, const PetscInt c, const PetscReal direction, const PetscInt *vertMask, PetscReal *g) {
  // The upwind direction is determined using the dot product between the vector u and the vector connecting the cell-center
  //    and the associated vertices

  const PetscInt    dim = subDomain->GetDimensions();
  PetscReal         weightTotal = 0.0;
  PetscScalar       x0[3] = {0.0, 0.0, 0.0}, n[3] = {0.0, 0.0, 0.0};

  ablate::utilities::MathUtilities::NormVector(dim, g, n);

  DMPlexComputeCellGeometryFVM(cellDM, c, NULL, x0, NULL) >> ablate::utilities::PetscUtilities::checkError;

  for (PetscInt d = 0; d < dim; ++d) {
    g[d] = 0.0;
  }

  // Obtain all cells which use this vertex
  PetscInt nVert, *verts;
  DMPlexCellGetVertices(cellDM, c, &nVert, &verts) >> ablate::utilities::PetscUtilities::checkError;

  for (PetscInt v = 0; v < nVert; ++v) {
    PetscReal x[3];
    DMPlexComputeCellGeometryFVM(cellDM, verts[v], NULL, x, NULL) >> ablate::utilities::PetscUtilities::checkError;

    ablate::utilities::MathUtilities::Subtract(dim, x0, x, x);
    ablate::utilities::MathUtilities::NormVector(dim, x, x);
    PetscReal dot = ablate::utilities::MathUtilities::DotVector(dim, n, x);

    dot *= direction;

    if (dot>0.0) {

      weightTotal += dot;

      const PetscInt id = reverseVertList[verts[v]];

      // Weighted average of the surrounding cell-center gradients.
      //  Note that technically this is (in 2D) the area of the quadrilateral that is formed by connecting
      //  the vertex, center of the neighboring edges, and the center of the triangle. As the three quadrilaterals
      //  that are formed this way all have the same area, there is no need to take into account the 1/3. Something
      //  similar should hold in 3D and for other cell types that ABLATE uses.
      for (PetscInt d = 0; d < dim; ++d) {
        g[d] += dot*gradArray[id*dim + d];
      }
    }
  }

  DMPlexCellRestoreVertices(cellDM, c, &nVert, &verts) >> ablate::utilities::PetscUtilities::checkError;

  // Size of the communicator
//  MPI_Comm comm = PetscObjectComm((PetscObject)dm);
//  int size;
//  MPI_Comm_size(comm, &size) >> ablate::utilities::MpiUtilities::checkError;

  // Error checking
  if ( PetscAbs(weightTotal) < ablate::utilities::Constants::small ) {
    // When running on a single processor all vertices should have an upwind cell. Throw an error if that's not the case.
    // When running in parallel, ghost vertices at the edge of the local domain may not have any surrounding upwind cells, so
    //  ignore the error and simply set the upwind gradient to zero.
    for (PetscInt d = 0; d < dim; ++d) {
      g[d] = 0.0;
    }
  }
  else {
    for (PetscInt d = 0; d < dim; ++d) {
      g[d] /= weightTotal;
    }
  }
}

void Reconstruction::Extension(const PetscInt *cellMask, const PetscInt *vertMask, Vec lsVec[2], Vec fVec[2]) {

  const PetscInt  dim = subDomain->GetDimensions();
  PetscReal       maxDiff = 1.0;
  PetscInt        iter = 0;
  PetscReal       *vertGrad = nullptr, *lsGrad = nullptr, *cellSign;
  PetscReal       h = 0.0;
  MPI_Comm        cellCOMM = PetscObjectComm((PetscObject)cellDM);
  PetscInt        *upwindCell = nullptr;

  const PetscInt  cellMaskRange = 4; // Maximum cell-mask ID to extend to

  DMPlexGetMinRadius(cellDM, &h) >> ablate::utilities::PetscUtilities::checkError;
  h *= 2.0; // Min radius returns the distance between a cell-center and a face. Double it to get the average cell size


  DMGetWorkArray(cellDM, nTotalCell, MPIU_INT, &upwindCell) >> ablate::utilities::PetscUtilities::checkError;
  DMGetWorkArray(cellDM, nTotalCell, MPIU_REAL, &cellSign) >> ablate::utilities::PetscUtilities::checkError;
  DMGetWorkArray(cellGradDM, nTotalCell*dim, MPIU_REAL, &lsGrad) >> ablate::utilities::PetscUtilities::checkError;
  DMGetWorkArray(vertGradDM, nTotalVert*dim, MPIU_REAL, &vertGrad) >> ablate::utilities::PetscUtilities::checkError;

  const PetscScalar *lsArray;
  VecGetArrayRead(lsVec[LOCAL], &lsArray);
  for (PetscInt c = 0; c < nTotalCell; ++c) {
    if (cellMask[c] > 0) {
      const PetscInt cell = cellList[c];
      DMPlexCellGradFromVertex(vertDM, cellList[c], lsVec[LOCAL], -1, 0, &lsGrad[c*dim]) >> ablate::utilities::PetscUtilities::checkError;

      // Get the sign of the level-set at the cell-center via averaging
      cellSign[c] = 0.0;
      PetscInt nVert, *verts;
      DMPlexCellGetVertices(cellDM, cell, &nVert, &verts);
      for (PetscInt v = 0; v < nVert; ++v) cellSign[c] += lsArray[vertList[v]];
      cellSign[c] /= nVert;
      DMPlexCellRestoreVertices(cellDM, cell, &nVert, &verts);
    }
  }
  VecRestoreArrayRead(lsVec[LOCAL], &lsArray);

  for (PetscInt c = 0; c < nTotalCell; ++c) {
    if (cellMask[c] > 0) {
      cellSign[c] = cellSign[c]/PetscSqrtReal(PetscSqr(cellSign[c]) + h*h);
    }
  }

  const PetscInt maxIter = 150;

  while (maxDiff>1.e-3 && iter<maxIter) {
    ++iter;

    // Determine the current gradient at vertices that need updating
    for (PetscInt v = 0; v < nTotalVert; ++v) {
      if (vertMask[v] > 0) {
        DMPlexVertexGradFromCell(cellDM, vertList[v], fVec[LOCAL], -1, 0, &vertGrad[v*dim]) >> ablate::utilities::PetscUtilities::checkError;
      }
    }

    maxDiff = -PETSC_MAX_REAL;

    PetscScalar *fArray = nullptr;
    VecGetArray(fVec[GLOBAL], &fArray) >> utilities::PetscUtilities::checkError;
    for (PetscInt c = 0; c < nLocalCell; ++c) {

      if (cellMask[c] > 1 && cellMask[c] < cellMaskRange) {
        const PetscInt cell = cellList[c];

        PetscReal g[dim];
        for (PetscInt d = 0; d < dim; ++d) g[d] = lsGrad[c*dim + d];

        CellUpwind(vertGrad, cell, PetscSignReal(cellSign[c]), g);

        PetscReal dH = 0.0;
        for (PetscInt d = 0; d < dim; ++d) dH += g[d]*lsGrad[c*dim + d];

        fArray[c] -= 0.5*h*cellSign[c]*dH;

        maxDiff = PetscMax(maxDiff, PetscAbsReal(dH));
      }
    }
    VecRestoreArray(fVec[GLOBAL], &fArray) >> utilities::PetscUtilities::checkError;
    DMGlobalToLocal(cellDM, fVec[GLOBAL], INSERT_VALUES, fVec[LOCAL]) >> utilities::PetscUtilities::checkError;

     // Get the maximum change across all processors. This also acts as a sync point
    MPI_Allreduce(MPI_IN_PLACE, &maxDiff, 1, MPIU_REAL, MPIU_MAX, cellCOMM);

//    VecGetArray(fVec[LOCAL], &fArray) >> utilities::PetscUtilities::checkError;
//    for (PetscInt c = 0; c < nTotalCell; ++c) {
//      if (cellMask[c] == cellMaskRange) {
//        const PetscInt cell = cellList[c];
//        PetscInt nCells, *cells;
//        DMPlexGetNeighbors(cellDM, cell, 1, -1.0, -1, PETSC_FALSE, PETSC_FALSE, &nCells, &cells);
//        PetscInt nv = 0;
//        fArray[c] = 0.0;
//        for (PetscInt nc = 0; nc < nCells; ++nc) {
//          const PetscInt id = reverseCellList[cells[nc]];
//          if (cellMask[id] == cellMaskRange-1 ) {
//            fArray[c] += fArray[id];
//            ++nv;
//          }
//        }
//        fArray[c] /= nv;
//        DMPlexRestoreNeighbors(cellDM, cell, 1, -1.0, -1, PETSC_FALSE, PETSC_FALSE, &nCells, &cells);
//      }
//    }
//    VecRestoreArray(fVec[LOCAL], &fArray) >> utilities::PetscUtilities::checkError;

    PetscPrintf(PETSC_COMM_WORLD, "Extension %3" PetscInt_FMT": %e\n", iter, maxDiff);
  }

  DMRestoreWorkArray(cellDM, nLocalCell, MPIU_REAL, &cellSign) >> ablate::utilities::PetscUtilities::checkError;
  DMRestoreWorkArray(cellGradDM, nLocalCell*dim, MPIU_REAL, &lsGrad) >> ablate::utilities::PetscUtilities::checkError;
  DMRestoreWorkArray(vertGradDM, nTotalVert*dim, MPIU_REAL, &vertGrad) >> ablate::utilities::PetscUtilities::checkError;

}


void Reconstruction::ToLevelSet(DM vofDM, Vec vofVec, const ablate::domain::Field vofField) {

int rank, size;
MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
MPI_Comm_size(PETSC_COMM_WORLD, &size);

for (PetscInt memIter = 0; memIter<1; ++memIter) {

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
  DMGetLocalVector(cellDM, &smoothVOFVec[LOCAL]) >> utilities::PetscUtilities::checkError;
  DMGetGlobalVector(cellDM, &smoothVOFVec[GLOBAL]) >> utilities::PetscUtilities::checkError;
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


  SetMasks(nLevels, cellMask, vertMask, smoothVOFVec);

SaveData(cellDM, cellMask, nTotalCell, cellList, "cellMask.txt", 1);
SaveData(vertDM, vertMask, nTotalVert, vertList, "vertMask.txt", 1);


  InitalizeLevelSet(smoothVOFVec[LOCAL], cellMask, vertMask, lsVec);



SaveData(vertDM, lsVec[LOCAL], nTotalVert, vertList, "vertLS0_L.txt", 1);
SaveData(vertDM, lsVec[GLOBAL], nLocalVert, vertList, "vertLS0_G.txt", 1);

  ReinitializeLevelSet(cellMask, vertMask, lsVec);

SaveData(vertDM, lsVec[LOCAL], nTotalVert, vertList, "vertLS1_L.txt", 1);
SaveData(vertDM, lsVec[GLOBAL], nLocalVert, vertList, "vertLS1_G.txt", 1);
xexit("");
  Vec curv[2];
  DMGetLocalVector(cellDM, &curv[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
  DMGetGlobalVector(cellDM, &curv[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;


  CalculateCellCurvatures(cellMask, vertMask, lsVec, curv);
SaveData(cellDM, curv[LOCAL], nLocalCell, cellList, "curv0.txt", 1);

  Extension(cellMask, vertMask, lsVec, curv);
SaveData(cellDM, curv[LOCAL], nLocalCell, cellList, "curv1.txt", 1);
xexit("");


  DMRestoreLocalVector(cellDM, &curv[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
  DMRestoreGlobalVector(cellDM, &curv[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;
  DMRestoreWorkArray(vertDM, nTotalVert, MPIU_INT, &vertMask) >> ablate::utilities::PetscUtilities::checkError;




//  DMRestoreLocalVector(cellGradDM, &cellGradVec[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
//  DMRestoreGlobalVector(cellGradDM, &cellGradVec[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;

  DMRestoreLocalVector(cellDM, &smoothVOFVec[LOCAL]) >> utilities::PetscUtilities::checkError;
  DMRestoreGlobalVector(cellDM, &smoothVOFVec[GLOBAL]) >> utilities::PetscUtilities::checkError;

  if (subpointIndices) ISRestoreIndices(subpointIS, &subpointIndices) >> utilities::PetscUtilities::checkError;

}
//xexit("");

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






}


