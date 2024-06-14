#include "interfaceReconstruction.hpp"
#include <petsc.h>
#include "utilities/petscSupport.hpp"
#include "utilities/mathUtilities.hpp"
#include "domain/fieldAccessor.hpp"
#include "levelSetUtilities.hpp"
#include "utilities/constants.hpp"
#include <petscblaslapack.h>




using namespace ablate::levelSet;

#define xexit(S, ...) {PetscFPrintf(MPI_COMM_WORLD, stderr, \
  "\x1b[1m(%s:%d, %s)\x1b[0m\n  \x1b[1m\x1b[90mexiting:\x1b[0m " S "\n", \
  __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__); exit(0);}

PetscBool IsPoint(DM dm, const PetscInt p, const PetscReal x, const PetscReal y) {
  PetscReal x0[3];

  DMPlexComputeCellGeometryFVM(dm, p, NULL, x0, NULL) >> ablate::utilities::PetscUtilities::checkError;

  return (PetscBool)((PetscAbsReal(x - x0[0]) < 1e-4) && (PetscAbsReal(y - x0[1]) < 1e-4));

}

static PetscInt FindCell(DM dm, const PetscInt dim, const PetscReal x0[], const PetscInt nCells, const PetscInt cells[]) {
  // Return the cell with the cell-center that is the closest to a given point
  for (PetscInt c = 0; c < nCells; ++c) {
    PetscBool inCell = PETSC_FALSE;
    DMPlexInCell(dm, cells[c], x0, &inCell) >> ablate::utilities::PetscUtilities::checkError;
    if (inCell) return cells[c];
  }

  return -1;


}

void Reconstruction::BuildInterpGaussianList() {

  PetscReal h;
  DMPlexGetMinRadius(cellDM, &h) >> ablate::utilities::PetscUtilities::checkError;
  h *= 2.0; // Min radius returns the distance between a cell-center and a face. Double it to get the average cell size
  const PetscReal sigma = sigmaFactor*h;

  PetscInt dim;
  DMGetDimension(cellDM, &dim) >> ablate::utilities::PetscUtilities::checkError;

  nGaussStencil = PetscPowInt(gaussianNQuad, dim); // The number of cells in the integration stencil

  const PetscInt nGaussRange[3] = {gaussianNQuad, (dim > 1) ? gaussianNQuad : 1, (dim > 2) ? gaussianNQuad : 1};

  PetscMalloc1(nGaussStencil*nTotalCell, &interpGaussianList) >> ablate::utilities::PetscUtilities::checkError;

  for (PetscInt c = 0; c < nTotalCell; ++c) {

    const PetscInt cell = cellList[c];

    PetscReal x0[3] = {0.0, 0.0, 0.0};
    DMPlexComputeCellGeometryFVM(cellDM, cell, NULL, x0, NULL) >> ablate::utilities::PetscUtilities::checkError;

    PetscInt nCells, *cellList;
    DMPlexGetNeighbors(cellDM, cell, 3, -1.0, -1, PETSC_FALSE, PETSC_FALSE, &nCells, &cellList) >> ablate::utilities::PetscUtilities::checkError;

    for (PetscInt i = 0; i < nGaussRange[0]; ++i) {
      for (PetscInt j = 0; j < nGaussRange[1]; ++j) {
        for (PetscInt k = 0; k < nGaussRange[2]; ++k) {

          PetscReal x[3] = {x0[0] + sigma*gaussianQuad[i], x0[1] + sigma*gaussianQuad[j], x0[2] + sigma*gaussianQuad[k]};

          const PetscInt interpCell = FindCell(cellDM, dim, x, nCells, cellList);

//          if (interpCell < 0) {
//            int rank;
//            MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
//            throw std::runtime_error("BuildInterpCellList could not determine the location of (" + std::to_string(x[0]) + ", " + std::to_string(x[1]) + ", " + std::to_string(x[2]) + ") on rank " + std::to_string(rank) + ".");
//          }

          interpGaussianList[c*nGaussStencil + gaussianNQuad*(k*gaussianNQuad + j) + i] = interpCell;
        }
      }
    }

    DMPlexRestoreNeighbors(cellDM, cell, 3, -1.0, -1, PETSC_FALSE, PETSC_FALSE, &nCells, &cellList) >> ablate::utilities::PetscUtilities::checkError;

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
  //    right away.
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
  DM              subAuxDM = subDomain->GetSubAuxDM();

  DMPlexGetMinRadius(subDomain->GetDM(), &h) >> ablate::utilities::PetscUtilities::checkError;
  h *= 2.0; // Min r

  // Setup the RBF interpolants
  vertRBF = std::make_shared<ablate::domain::rbf::IMQ>(polyAug, 1e-2*h, doesNotHaveDerivatives, doesNotHaveInterpolation, true);
  vertRBF->Setup(subDomain);
  vertRBF->Initialize();


  cellRBF = std::make_shared<ablate::domain::rbf::IMQ>(polyAug, 1e-2*h, doesNotHaveDerivatives, doesNotHaveInterpolation, false);
  cellRBF->Setup(subDomain);
  cellRBF->Initialize();

  convolution = std::make_shared<ablate::levelSet::GaussianConvolution>(subAuxDM, 3, 1.0);

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
  BuildInterpGaussianList();

}

Reconstruction::~Reconstruction() {

xexit("");


  PetscFree(interpGaussianList) >> ablate::utilities::PetscUtilities::checkError;

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

void Reconstruction_SaveDM(DM dm, const char fname[255]) {

  int rank, size;
  MPI_Comm_size(PETSC_COMM_WORLD, &size);
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  PetscInt dim;

  DMGetDimension(dm, &dim);

  PetscInt eStart, eEnd;
  DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd) >> ablate::utilities::PetscUtilities::checkError;
  for (PetscInt r = 0; r < size; ++r) {
    if ( rank==r ) {

      FILE *f1;
      if ( rank==0 ) f1 = fopen(fname, "w");
      else f1 = fopen(fname, "a");

      for (PetscInt e = eStart; e < eEnd; ++e) {
        PetscInt nVert;
        DMPlexGetConeSize(dm, e, &nVert);

        if (nVert==2) {
          const PetscInt *verts;
          DMPlexGetCone(dm, e, &verts);
          PetscReal x0[3], x1[3];
          DMPlexComputeCellGeometryFVM(dm, verts[0], NULL, x0, NULL) >> ablate::utilities::PetscUtilities::checkError;
          DMPlexComputeCellGeometryFVM(dm, verts[1], NULL, x1, NULL) >> ablate::utilities::PetscUtilities::checkError;

          fprintf(f1, "plot([%f %f],[%f %f],'k-');\n", x0[0], x1[0], x0[1], x1[1]);
        }

      }
      fclose(f1);
    }
  }


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


////   Turn off any "cut cells" where the cell is not surrounded by any other cut cells.
////   To avoid cut-cells two cells-thick turn off any cut-cells which have a neighoring gradient passing through them.
//  const PetscInt    dim = subDomain->GetDimensions();
//  for (PetscInt c = 0; c < nTotalCell; ++c) {

//    if (cellMask[c] == 1) {

//      const PetscInt cell = cellList[c];

//      PetscInt nCells, *cells;
//      DMPlexGetNeighbors(cellDM, cell, 1, -1.0, -1, PETSC_FALSE, PETSC_FALSE, &nCells, &cells) >> ablate::utilities::PetscUtilities::checkError;
//      PetscInt nCut = 0;
//      for (PetscInt i = 0; i < nCells; ++i) {
//        PetscInt id = reverseCellList[cells[i]];
//        nCut += (cellMaskVecArray[id] > 0.5);
//      }
//      DMPlexRestoreNeighbors(cellDM, cell, 1, -1.0, -1, PETSC_FALSE, PETSC_FALSE, &nCells, &cells) >> ablate::utilities::PetscUtilities::checkError;

//      cellMask[c] = (nCut>1); // If nCut equals 1 then the center cell is the only cut cell, so deactivate it

//      PetscScalar n[dim];
//      DMPlexCellGradFromCell(cellDM, cell, vofVec[LOCAL], -1, 0, n) >> ablate::utilities::PetscUtilities::checkError;

//      if (cellMask[c]==1 && ablate::utilities::MathUtilities::MagVector(dim, n)>PETSC_SMALL) {
//        // Now check for two-deep cut-cells.
//        const PetscReal direction[2] = {-1.0, +1.0};
//        for (PetscInt d = 0; d < 2; ++d) {
//          PetscInt neighborCell = -1;
//          DMPlexGetForwardCell(cellDM, cell, n, direction[d], &neighborCell) >> ablate::utilities::PetscUtilities::checkError;
//          if (neighborCell > -1) {
//            neighborCell = reverseCellList[neighborCell];

//            if (PetscAbsReal(vofArray[neighborCell] - 0.5) < PetscAbsReal(vofArray[c] - 0.5)) {
//              cellMask[c] = 0;
//              break;
//            }
//          }
//        }
//      }
//    }
//  }

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
void Reconstruction::InitalizeLevelSet(Vec vofVec, const PetscInt *cellMask, const PetscInt *vertMask, Vec lsVec[2], PetscReal *closestPoint, PetscInt *cpCell) {

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

  while ( maxDiff > 1e-3*h && iter<2000 ) {

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

    PetscPrintf(PETSC_COMM_WORLD, "Cut Cells %" PetscInt_FMT": %+e\n", iter, maxDiff) >> ablate::utilities::PetscUtilities::checkError;

  }

  if (maxDiff > 1e-3*h) {
    throw std::runtime_error("Interface reconstruction has failed.\n");
  }

FILE *f1 = fopen("cellGrad.txt", "w");
for (PetscInt c = 0; c < nTotalCell; ++c) {
  if (cellMask[c] == 1) {
    PetscInt cell = cellList[c];
    PetscReal x[3];
    DMPlexComputeCellGeometryFVM(cellDM, cell, NULL, x, NULL) >> ablate::utilities::PetscUtilities::checkError;
    fprintf(f1, "%+e\t%+e\t%+e\t%+e\n", x[0], x[1], cellGrad[c*dim+0], cellGrad[c*dim+1]);
  }
}
fclose(f1);

f1 = fopen("vertLS.txt","w");
PetscScalar *lsVal = nullptr;
VecGetArray(lsVec[LOCAL], &lsVal) >> ablate::utilities::PetscUtilities::checkError;
for (PetscInt v = 0; v < nTotalVert; ++v) {
  if (vertMask[v] == 1) {
    PetscInt vert = vertList[v];
    PetscReal x[3];
    DMPlexComputeCellGeometryFVM(vertDM, vert, NULL, x, NULL) >> ablate::utilities::PetscUtilities::checkError;
    fprintf(f1, "%+e\t%+e\t%+e\n", x[0], x[1], lsVal[v]);
  }
}
fclose(f1);



  // Get the closest point to each vertex, assuming a linear interface in a cell.
  PetscReal *cellPhi;
  DMGetWorkArray(cellDM, nLocalCell, MPIU_REAL, &cellPhi) >> ablate::utilities::PetscUtilities::checkError;

  PetscArrayzero(cellPhi, nLocalCell) >> ablate::utilities::PetscUtilities::checkError;
  const PetscScalar *lsArray = nullptr;
  VecGetArrayRead(lsVec[LOCAL], &lsArray) >> ablate::utilities::PetscUtilities::checkError;
  for (PetscInt c = 0; c < nLocalCell; ++c) {
    if (cellMask[c] == 1) {
      const PetscInt cell = cellList[c];

      PetscInt nVerts, *verts;
      DMPlexCellGetVertices(cellDM, cell, &nVerts, &verts) >> utilities::PetscUtilities::checkError;
      for (PetscInt v = 0; v < nVerts; ++v) {
        const PetscInt id = reverseVertList[verts[v]];
        cellPhi[c] += lsArray[id];
      }
      cellPhi[c] /= nVerts;
      DMPlexCellRestoreVertices(cellDM, cell, &nVerts, &verts) >> utilities::PetscUtilities::checkError;
    }
  }
  VecRestoreArrayRead(lsVec[LOCAL], &lsArray) >> ablate::utilities::PetscUtilities::checkError;

  for (PetscInt v = 0; v < nLocalVert; ++v) {
    if (vertMask[v] == 1) {
      const PetscInt vert = vertList[v];
      PetscReal x0[3];
      DMPlexComputeCellGeometryFVM(vertDM, vert, NULL, x0, NULL) >> ablate::utilities::PetscUtilities::checkError;

      PetscInt nCells, *cells;
      DMPlexVertexGetCells(vertDM, vert, &nCells, &cells) >> ablate::utilities::PetscUtilities::checkError;

      PetscReal minDist = PETSC_MAX_REAL;

      for (PetscInt c = 0; c < nCells; ++c) {
        const PetscInt id = reverseCellList[cells[c]];
        if (cellMask[id] == 1) {
          PetscReal x[3];
          DMPlexComputeCellGeometryFVM(vertDM, cells[c], NULL, x, NULL) >> ablate::utilities::PetscUtilities::checkError;


          PetscReal cv[dim];
          PetscReal dot = 0.0;
          const PetscReal *n = &cellGrad[id*dim];
          for (PetscInt d = 0; d < dim; ++d){
            x[d] -= cellPhi[id] * n[d]; // Shifted "center" of the plane from the cell-center
            cv[d] = x0[d] - x[d];       // Vector from the "center" of the plane to the vertex
            dot  += cv[d] * n[d];
          }

          if (PetscAbsReal(dot) < minDist) {
            minDist = PetscAbsReal(dot);
            cpCell[v] = id;
            for (PetscInt d = 0; d < dim; ++d) {
              closestPoint[v*dim + d] = x[d] + cv[d] - dot*n[d];
            }
          }
        }
      }
      DMPlexVertexRestoreCells(vertDM, vert, &nCells, &cells) >> ablate::utilities::PetscUtilities::checkError;
    }
    else {
      cpCell[v] = -1;
      for (PetscInt d = 0; d < dim; ++d) closestPoint[v*dim + d] = 0.0;
    }
  }
  DMRestoreWorkArray(cellDM, nLocalCell, MPIU_REAL, &cellPhi) >> ablate::utilities::PetscUtilities::checkError;


  DMRestoreWorkArray(vertDM, nLocalVert, MPIU_INT, &lsCount) >> ablate::utilities::PetscUtilities::checkError;
  DMRestoreWorkArray(cellGradDM, nTotalCell*dim, MPIU_REAL, &cellGrad) >> ablate::utilities::PetscUtilities::checkError;



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

  PetscScalar *lsGlobalArray = nullptr;
  VecGetArray(lsVec[GLOBAL], &lsGlobalArray) >> ablate::utilities::PetscUtilities::checkError;
  for (PetscInt c = 0 ; c < nLocalCell; ++c) {
    const PetscInt cell = cellList[c];
    PetscInt nVerts, *verts;

    DMPlexCellGetVertices(vertDM, cell, &nVerts, &verts) >> utilities::PetscUtilities::checkError;

    const PetscReal lsSetValues[2] = {lsRange[ vofArray[c] < 0.5 ? 1 : 0 ], PetscSignReal(0.5 - vofArray[c])*maxDist};
//    const PetscReal cellSign = (vofArray[c] < 0.5 ? +1.0 : -1.0);

    for (PetscInt v = 0; v < nVerts; ++v) {
      const PetscInt id = reverseVertList[verts[v]];

      if (id < nLocalVert) {

        if(vertMask[id] > 1) lsGlobalArray[id] = lsSetValues[0];// + cellSign*(vertMask[id] - 0.5)*h;
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
  const PetscInt maxIter = 250;

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

        DMPlexVertexGradFromVertex(vertDM, vertList[v], lsVec[LOCAL], -1, 0, &vertGrad[v*dim]) >> ablate::utilities::PetscUtilities::checkError;
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

          lsArray[v] -= h*sgn*(nrm - 1.0);


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

void Reconstruction::CalculateVertexCurvatures(const PetscInt *cellMask, const PetscInt *vertMask, Vec lsVec[2], PetscReal *closestPoint, PetscInt *cpCell, Vec curvVec[2]) {

  const PetscInt    dim = subDomain->GetDimensions();
  PetscReal h;
  DMPlexGetMinRadius(vertDM, &h) >> ablate::utilities::PetscUtilities::checkError;
  h *= 2.0;

  Vec phiGrad[2];
  DMGetLocalVector(cellGradDM, &phiGrad[LOCAL]) >> utilities::PetscUtilities::checkError;
  DMGetGlobalVector(cellGradDM, &phiGrad[GLOBAL]) >> utilities::PetscUtilities::checkError;

  VecZeroEntries(phiGrad[GLOBAL]) >> utilities::PetscUtilities::checkError;

  const PetscInt dx[3] = {1, 0, 0}, dy[3] = {0, 1, 0}, dz[3] = {0, 0, 1};
  PetscScalar *gradArray;
  VecGetArray(phiGrad[GLOBAL], &gradArray);
  for (PetscInt c = 0; c < nLocalCell; ++c) {
    if (cellMask[c] > 0) {
      const PetscInt cell = cellList[c];

      if (cellMask[c] < nLevels-2) {
        convolution->Evaluate(vertDM, vertRBF, -1, lsVec[LOCAL], 0, cell, dim, dx, dy, dz, &gradArray[c*dim]);
      }
      else {
        DMPlexCellGradFromVertex(vertDM, cell, lsVec[LOCAL], -1, 0, &gradArray[c*dim]) >> utilities::PetscUtilities::checkError;
      }
      ablate::utilities::MathUtilities::NormVector(dim, &gradArray[c*dim], &gradArray[c*dim]);
    }
  }
  VecRestoreArray(phiGrad[GLOBAL], &gradArray) >> utilities::PetscUtilities::checkError;


  DMGlobalToLocal(cellGradDM, phiGrad[GLOBAL], INSERT_VALUES, phiGrad[LOCAL]) >> utilities::PetscUtilities::checkError;
  DMRestoreGlobalVector(cellGradDM, &phiGrad[GLOBAL]) >> utilities::PetscUtilities::checkError;



  PetscScalar *array;
  VecGetArray(curvVec[GLOBAL], &array) >> utilities::PetscUtilities::checkError;

  for (PetscInt v = 0; v < nLocalVert; ++v) {
    array[v] = 0.0;
    if (vertMask[v] > 0) {
      const PetscInt vert = vertList[v];

      if (vertMask[v] < nLevels-2) {
        PetscReal val;
        for (PetscInt d = 0; d < dim; ++d) {
          convolution->Evaluate(cellGradDM, cellRBF, -1, phiGrad[LOCAL], d, vert, 1, &dx[d], &dy[d], &dz[d], &val);
          array[v] += val;
        }
      }
      else if (vertMask[v] == nLevels-2){
        PetscReal g[dim];
        for (PetscInt d = 0; d < dim; ++d) {
          DMPlexVertexGradFromCell(cellGradDM, vert, phiGrad[LOCAL], -1, d, g) >> utilities::PetscUtilities::checkError;
          array[v] += g[d];
        }
      }

    }
  }
  VecRestoreArray(curvVec[GLOBAL], &array) >> utilities::PetscUtilities::checkError;
  DMRestoreLocalVector(cellGradDM, &phiGrad[LOCAL]) >> utilities::PetscUtilities::checkError;
  DMGlobalToLocal(vertDM, curvVec[GLOBAL], INSERT_VALUES, curvVec[LOCAL]) >> utilities::PetscUtilities::checkError;
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


// Extension of vertex-based data
void Reconstruction::Smooth(const PetscInt *cellMask, const PetscInt *vertMask, Vec lsVec[2], Vec fVec[2]) {

  const PetscInt  dim = subDomain->GetDimensions();
  PetscReal       *lsGrad = nullptr;
  PetscReal       h = 0.0;
  Vec             fGradVec;

  DMPlexGetMinRadius(cellDM, &h) >> ablate::utilities::PetscUtilities::checkError;
  h *= 2.0; // Min radius returns the distance between a cell-center and a face. Double it to get the average cell size


  DMGetWorkArray(cellGradDM, nTotalCell*dim, MPIU_REAL, &lsGrad) >> ablate::utilities::PetscUtilities::checkError;
  DMGetLocalVector(cellGradDM, &fGradVec);

  for (PetscInt c = 0; c < nTotalCell; ++c) {
    if (cellMask[c] > 0) {
      const PetscInt cell = cellList[c];
      DMPlexCellGradFromVertex(vertDM, cell, lsVec[LOCAL], -1, 0, &lsGrad[c*dim]) >> ablate::utilities::PetscUtilities::checkError;
      ablate::utilities::MathUtilities::NormVector(dim, &lsGrad[c*dim]);
    }
  }

  for (PetscInt iter = 0; iter < 100; ++iter) {

    PetscScalar *fGrad = nullptr;
    VecGetArray(fGradVec, &fGrad);
    for (PetscInt c = 0; c < nTotalCell; ++c) {
      if (cellMask[c] > 0) {
        const PetscInt cell = cellList[c];
        DMPlexCellGradFromVertex(vertDM, cell, fVec[LOCAL], -1, 0, &fGrad[c*dim]) >> ablate::utilities::PetscUtilities::checkError;
        PetscReal nrm = ablate::utilities::MathUtilities::DotVector(dim, &lsGrad[c*dim], &fGrad[c*dim]);
        for (PetscInt d = 0; d < dim; ++d) fGrad[c*dim + d] -= lsGrad[c*dim + d]*nrm;
      }
    }
    VecRestoreArray(fGradVec, &fGrad);


    PetscScalar *fArray = nullptr;
    VecGetArray(fVec[GLOBAL], &fArray) >> utilities::PetscUtilities::checkError;
    const PetscScalar *lsArray = nullptr;
    VecGetArrayRead(lsVec[GLOBAL], &lsArray) >> utilities::PetscUtilities::checkError;

    for (PetscInt v = 0; v < nLocalVert; ++v) {
      if (vertMask[v] > 0 && vertMask[v] < 4) {
        const PetscInt vert = vertList[v];

        PetscReal g[dim], div = 0.0;
        for (PetscInt d = 0; d < dim; ++d){
          DMPlexVertexGradFromCell(cellGradDM, vert, fGradVec, -1, d, g) >> ablate::utilities::PetscUtilities::checkError;
          div += g[d];
        }

        fArray[v] += 0.5*h*h*div;
      }
    }
    VecRestoreArrayRead(lsVec[GLOBAL], &lsArray) >> utilities::PetscUtilities::checkError;
    VecRestoreArray(fVec[GLOBAL], &fArray) >> utilities::PetscUtilities::checkError;
    DMGlobalToLocal(vertDM, fVec[GLOBAL], INSERT_VALUES, fVec[LOCAL]) >> utilities::PetscUtilities::checkError;
  }

  DMRestoreWorkArray(cellGradDM, nTotalCell*dim, MPIU_REAL, &lsGrad) >> ablate::utilities::PetscUtilities::checkError;
  DMRestoreLocalVector(cellGradDM, &fGradVec) >> ablate::utilities::PetscUtilities::checkError;

}

// Extension of vertex-based data
void Reconstruction::Extension(const PetscInt *cellMask, const PetscInt *vertMask, Vec lsVec[2], PetscReal *closestPoint, PetscInt *cpCell, Vec fVec[2]) {

  const PetscInt  dim = subDomain->GetDimensions();
  PetscReal       maxDiff = 1.0;
  PetscInt        iter = 0;
  PetscReal       *lsGrad = nullptr, *cellGrad = nullptr;
  Vec cellGradVec;
  PetscReal       h = 0.0;
  MPI_Comm        cellCOMM = PetscObjectComm((PetscObject)cellDM);

  DMPlexGetMinRadius(cellDM, &h) >> ablate::utilities::PetscUtilities::checkError;
  h *= 2.0; // Min radius returns the distance between a cell-center and a face. Double it to get the average cell size


  DMGetWorkArray(vertGradDM, nTotalVert*dim, MPIU_REAL, &lsGrad) >> ablate::utilities::PetscUtilities::checkError;
//  DMGetWorkArray(cellGradDM, nTotalCell*dim, MPIU_REAL, &cellGrad) >> ablate::utilities::PetscUtilities::checkError;
  DMGetLocalVector(cellGradDM, &cellGradVec);

  VecGetArray(cellGradVec, &cellGrad);

  for (PetscInt v = 0; v < nTotalVert; ++v) {
    if (vertMask[v] > 0) {
      const PetscInt vert = vertList[v];
      DMPlexVertexGradFromVertex(vertDM, vert, lsVec[LOCAL], -1, 0, &lsGrad[v*dim]) >> ablate::utilities::PetscUtilities::checkError;
      ablate::utilities::MathUtilities::NormVector(dim, &lsGrad[v*dim]);

    }
  }

  const PetscInt maxIter = 250;

  while (maxDiff>1.e-3 && iter<maxIter) {
    ++iter;

    // Determine the current gradient at vertices that need updating
    for (PetscInt c = 0; c < nTotalCell; ++c) {
      if (cellMask[c] > 0) {
        DMPlexCellGradFromVertex(vertDM, cellList[c], fVec[LOCAL], -1, 0, &cellGrad[c*dim]) >> ablate::utilities::PetscUtilities::checkError;
      }
    }


    maxDiff = -PETSC_MAX_REAL;


    const PetscScalar *lsArray;
    PetscScalar *fArray = nullptr;

    VecGetArrayRead(lsVec[LOCAL], &lsArray) >> utilities::PetscUtilities::checkError;
    VecGetArray(fVec[GLOBAL], &fArray) >> utilities::PetscUtilities::checkError;
    for (PetscInt v = 0; v < nLocalVert; ++v) {
      if (vertMask[v] > 1 && vertMask[v] < 500) {
        const PetscInt vert = vertList[v];

        PetscReal g[dim];
        for (PetscInt d = 0; d < dim; ++d) g[d] = lsGrad[v*dim + d];

        VertexUpwind(cellGrad, vert, PetscSignReal(lsArray[v]), cellMask, g);


        PetscReal dH = 0.0;
        for (PetscInt d = 0; d < dim; ++d) dH += g[d]*lsGrad[v*dim + d];

//        PetscReal sgn = (lsArray[v])/PetscSqrtReal(PetscSqr(lsArray[v]) + PetscSqr(h));
        PetscReal sgn = PetscSignReal(lsArray[v]);
        fArray[v] -= h*sgn*dH;

        maxDiff = PetscMax(maxDiff, PetscAbsReal(dH));


//        if (vertMask[v] < nLevels - 2) {
//          PetscReal div = 0.0;
//          for (PetscInt d = 0; d < dim; ++d) {
//            PetscReal g[dim];
//            DMPlexVertexGradFromCell(cellGradDM, vert, cellGradVec, -1, d, g);
//            div += g[d];
//          }
//          fArray[v] += 0.1*h*h*div;
//        }

      }
//      else if (vertMask[v] == 1) {
//        fArray[v] = vertRBF->Interpolate(vertDM, -1, fVec[LOCAL], cpCell[v], &closestPoint[v*dim]);
////          PetscInt zero[1] = {0};
////          convolution->Evaluate(vertDM, vertRBF, -1, fVec[LOCAL], 0, vertList[v], 1, zero, zero, zero, &fArray[v]);

//      }
    }

    VecRestoreArrayRead(lsVec[LOCAL], &lsArray) >> utilities::PetscUtilities::checkError;
    VecRestoreArray(fVec[GLOBAL], &fArray) >> utilities::PetscUtilities::checkError;
    DMGlobalToLocal(vertDM, fVec[GLOBAL], INSERT_VALUES, fVec[LOCAL]) >> utilities::PetscUtilities::checkError;

     // Get the maximum change across all processors. This also acts as a sync point
    MPI_Allreduce(MPI_IN_PLACE, &maxDiff, 1, MPIU_REAL, MPIU_MAX, cellCOMM);


    PetscPrintf(PETSC_COMM_WORLD, "Extension %3" PetscInt_FMT": %e\n", iter, maxDiff);
  }
  VecRestoreArray(cellGradVec, &cellGrad);
  DMRestoreLocalVector(cellGradDM, &cellGradVec);
  DMRestoreWorkArray(vertGradDM, nTotalVert*dim, MPIU_REAL, &lsGrad) >> ablate::utilities::PetscUtilities::checkError;
//  DMRestoreWorkArray(cellGradDM, nTotalCell*dim, MPIU_REAL, &cellGrad) >> ablate::utilities::PetscUtilities::checkError;

}



// This implements a FMM-like algorithm to determine a signed distance function given a set of vertices which already have
//  an initial level-set
//  Let a cell with nv-vertices have level-set values at nv-1 vertices. Call the unknown level-set at the last vertex phi.
//  Then it is possible to construct a cell-centered gradient as g = a*phi + b, where a contains the contribution of the unknown level set
//  value and b is the contribution from the nv-1 other vertices. Making g.g==1 results in a quadratic equation, similar to the standard FMM method.
//  For vertices which share multiple possible neighbor cells choose the smallest of the possible results
void Reconstruction::FMM(const PetscInt *cellMask, const PetscInt *vertMask, Vec lsVec[2]) {


  PetscInt vStart, vEnd;
  DMPlexGetDepthStratum(vertDM, 0, &vStart, &vEnd);   // Range of vertices


  PetscInt *updatedVertex;
  DMGetWorkArray(vertDM, nTotalVert, MPIU_INT, &updatedVertex) >> ablate::utilities::PetscUtilities::checkError;
  PetscArrayzero(updatedVertex, nTotalVert);

  PetscScalar *lsArray[2] = {nullptr, nullptr};
  VecGetArray(lsVec[GLOBAL], &lsArray[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;

  PetscInt dim;
  DMGetDimension(vertDM, &dim);

  for (PetscInt v = 0; v < nTotalVert; ++v) {
    if (vertMask[v]==1){
      updatedVertex[v] = 1;
    }
    else if (vertMask[v]>1){
       lsArray[GLOBAL][v] = PetscSignReal(lsArray[GLOBAL][v])*PETSC_MAX_REAL;
     }
  }

  for (PetscInt currentLevel = 2; currentLevel <= 12; ++currentLevel) {

    while (true) {
      for (PetscInt c = 0; c < nTotalCell; ++c) {
        if (cellMask[c]==currentLevel) {
          PetscInt cell = cellList[c];
          PetscInt nVert, *verts;
          DMPlexCellGetVertices(vertDM, cell, &nVert, &verts);

          PetscInt nUpdated = 0, vertID = -1;
          for (PetscInt v = 0; v < nVert; ++v) {
            PetscInt id = reverseVertList[verts[v]];
            if (updatedVertex[id]==1) {
              ++nUpdated;
            }
            else {
              vertID = verts[v];
            }
          }

          if (nUpdated + 1 == nVert) {
            // This will create the gradient vector a*phi + b where phi is the level set value to find
            PetscReal a[3] = {0.0, 0.0, 0.0}, b[3] = {0.0, 0.0, 0.0};

            PetscInt nFace;
            const PetscInt *faces;

            // Get all faces associated with the cell
            DMPlexGetConeSize(vertDM, cell, &nFace);
            DMPlexGetCone(vertDM, cell, &faces);
            for (PetscInt f = 0; f < nFace; ++f) {
                PetscReal N[3] = {0.0, 0.0, 0.0};
                DMPlexFaceCentroidOutwardAreaNormal(vertDM, cell, faces[f], NULL, N);

                // All points associated with this face
                PetscInt nClosure, *closure = NULL;
                DMPlexGetTransitiveClosure(vertDM, faces[f], PETSC_TRUE, &nClosure, &closure);

                PetscReal cnt = 0.0, ave = 0.0, vertCoeff = 0.0;
                for (PetscInt cl = 0; cl < nClosure * 2; cl += 2) {
                  if (closure[cl] >= vStart && closure[cl] < vEnd) {  // Only use the points corresponding to a vertex

                    const PetscInt id = reverseVertList[closure[cl]];
                    if (closure[cl]==vertID) {
                      if (updatedVertex[id]==1) throw std::runtime_error("How can this be possible?\n");
                      ++vertCoeff;
                    }
                    else {
                      if (updatedVertex[id]==0) throw std::runtime_error("How can this be possible?\n");
                      ave += lsArray[GLOBAL][id];
                    }

                    cnt += 1.0;
                  }
                }

                DMPlexRestoreTransitiveClosure(vertDM, faces[f], PETSC_TRUE, &nClosure, &closure);  // Restore the points

                // Function value at the face center
                ave /= cnt;
                vertCoeff /= cnt;
                for (PetscInt d = 0; d < dim; ++d) {
                    a[d] += vertCoeff * N[d];
                    b[d] += ave * N[d];
                }
            }

            PetscReal vol;
            DMPlexComputeCellGeometryFVM(vertDM, cell, &vol, NULL, NULL);
            for (PetscInt d = 0; d < dim; ++d) {
                a[d] /= vol;
                b[d] /= vol;
            }

            PetscReal p2 = ablate::utilities::MathUtilities::DotVector(dim, a, a);
            PetscReal p1 = 2.0*ablate::utilities::MathUtilities::DotVector(dim, a, b);
            PetscReal p0 = ablate::utilities::MathUtilities::DotVector(dim, b, b) - 1.0;

            PetscReal disc = p1*p1 - 4.0*p0*p2;

            if (disc >= 0.0) {

              disc = PetscSqrtReal(disc);

              PetscReal phi0, phi1;
              if (p1 < 0) {
                phi0 = 2.0*p0/(-p1 + disc);
                phi1 = (-p1 + disc)/(2.0*p2);
              }
              else {
                phi0 = (-p1 - disc)/(2.0*p2);
                phi1 = 2.0*p0/(-p1 - disc);
              }

              const PetscInt id = reverseVertList[vertID];
              if (phi0*lsArray[GLOBAL][id] >= 0.0 && phi1*lsArray[GLOBAL][id] >= 0.0) { // Both have the correct sign
                PetscReal newPhi = PetscAbsReal(phi0) > PetscAbsReal(phi1) ? phi0 : phi1;
                lsArray[GLOBAL][id] = PetscAbsReal(newPhi) < PetscAbsReal(lsArray[GLOBAL][id]) ? newPhi : lsArray[GLOBAL][id];
              }
              else if (phi0*lsArray[GLOBAL][id] >= 0.0) {
                lsArray[GLOBAL][id] = PetscAbsReal(phi0) < PetscAbsReal(lsArray[GLOBAL][id]) ? phi0 : lsArray[GLOBAL][id];
              }
              else if (phi1*lsArray[GLOBAL][id] >= 0.0) {
                lsArray[GLOBAL][id] = PetscAbsReal(phi1) < PetscAbsReal(lsArray[GLOBAL][id]) ? phi1 : lsArray[GLOBAL][id];
              }
              else {
                throw std::runtime_error("Incorrect sign?\n");
              }

              updatedVertex[id] = 2;
            }
          }
          DMPlexCellRestoreVertices(vertDM, cell, &nVert, &verts);
        }
      }

      PetscInt nUpdated = 0;
      for (PetscInt v = 0; v < nTotalVert; ++v) {
        if (updatedVertex[v]==2) {
          ++nUpdated;
          updatedVertex[v] = 1;
        }
      }
//  printf("%ld\n", nUpdated);
      if (nUpdated==0) break;

    }

    // In certain instances not all of the vertices will be updated. A case where this happens is the following:
    //  2  --  1  -- 1
    //  2  --  1  -- 1
    //  2A --  2B -- 1
    //  3  --  2  -- 1
    // where 1 are vertices associated with cut-cells, 2/2A/2B are neighboring vertices, and 3 is the next level
    // Vertex 2B will be updated, but as vertex 3 will not be updated vertex 2A will alwasy be associated with a cell
    // that has two or more unknowns.
    //
    // Therefore, use the method given in "Fast methods for the Eikonal and related Hamilton–Jacobi equations on unstructured meshes"
    // by Sethian and Vladimirsky to handle those nodes.
    //
    // Note: Why not do this for all of the nodes? In some cases there may be only one neighboring vertex that is "accepted" (label==1).
    //        Rather than do a 1D approximation do the cell-based algorithm above and use this to fix any problem vertices


    for (PetscInt v = 0; v < nTotalVert; ++v) {

      if (vertMask[v]==currentLevel && updatedVertex[v]==0) {

        PetscInt vert = vertList[v];

        PetscInt nVert, *neighborVerts;
        DMPlexGetNeighbors(vertDM, vert, 1, -1.0, -1, PETSC_TRUE, PETSC_TRUE, &nVert, &neighborVerts);

        PetscReal x0[dim];
        DMPlexComputeCellGeometryFVM(vertDM, vert, NULL, x0, NULL) >> ablate::utilities::PetscUtilities::checkError;

        PetscReal P[(nVert-1)*dim], a[nVert-1], b[nVert-1], rhs[2*(nVert-1)]; // Define the matrices using the maximum possible size

        char transpose = 'T';
        PetscBLASInt m, n = 0, nrhs = 2, info, worksize;
        PetscBLASIntCast(dim, &m) >> ablate::utilities::PetscUtilities::checkError;
        PetscBLASIntCast((nVert - 1)*dim, &worksize) >> ablate::utilities::PetscUtilities::checkError;
        double work[worksize];

        for (PetscInt nv = 0; nv < nVert; ++nv) {
          const PetscInt neighbor = neighborVerts[nv];
          const PetscInt id = reverseVertList[neighbor];

          if (updatedVertex[id]==1) {

            PetscReal x[dim];
            DMPlexComputeCellGeometryFVM(vertDM, neighbor, NULL, x, NULL) >> ablate::utilities::PetscUtilities::checkError;

            for (PetscInt d = 0; d < dim; ++d) {
              P[n*dim + d] = x[d] - x0[d];
            }
            PetscReal mag = ablate::utilities::MathUtilities::MagVector(dim, &P[n*dim]);
            ablate::utilities::MathUtilities::ScaleVector(dim, &P[n*dim], 1.0/mag);
            a[n] = -1.0/mag;
            b[n] = lsArray[GLOBAL][id]/mag;
            ++n;

          }
        }

        DMPlexRestoreNeighbors(vertDM, vert, 1, -1.0, -1, PETSC_TRUE, PETSC_TRUE, &nVert, &neighborVerts);

        PetscReal p0, p1, p2;

        // When the vertex is equal distance between two interfaces the gradient will be zero.
        //  In this case remove points one at a time until a non-zero gradient is calculated.
        while (n >= m) {
          for (PetscInt i = 0; i < n; ++i) {
            rhs[i] = a[i];
            rhs[i + n] = b[i];
          }

          LAPACKgels_(&transpose, &m, &n, &nrhs, P, &m, rhs, &n, work, &worksize, &info);
          if (info != 0) throw std::runtime_error("Bad argument to GELS");

          p2 = ablate::utilities::MathUtilities::DotVector(dim, &rhs[0], &rhs[0]);
          p0 = ablate::utilities::MathUtilities::DotVector(dim, &rhs[n], &rhs[n]);

          if (p0 > PETSC_SMALL && p2 > PETSC_SMALL) break;

          --n;

        }

        if (n < m) throw std::runtime_error("Number of valid neighbor vertices is less than the dimension.");

        p1 = 2.0*ablate::utilities::MathUtilities::DotVector(dim, &rhs[0], &rhs[n]);
        p0 -= 1.0;

        PetscReal disc = p1*p1 - 4.0*p0*p2;

        if (disc < 0.0) {
          printf("%ld\t%ld\n", v, currentLevel);
          throw std::runtime_error("Getting imaginary roots");
        }

        if (disc >= 0.0) {

          disc = PetscSqrtReal(disc);

          PetscReal phi0, phi1;
          if (p1 < 0) {
            phi0 = 2.0*p0/(-p1 + disc);
            phi1 = (-p1 + disc)/(2.0*p2);
          }
          else {
            phi0 = (-p1 - disc)/(2.0*p2);
            phi1 = 2.0*p0/(-p1 - disc);
          }

          if (phi0*lsArray[GLOBAL][v] >= 0.0 && phi1*lsArray[GLOBAL][v] >= 0.0) { // Both have the correct sign
            PetscReal newPhi = PetscAbsReal(phi0) > PetscAbsReal(phi1) ? phi0 : phi1;
            lsArray[GLOBAL][v] = PetscAbsReal(newPhi) < PetscAbsReal(lsArray[GLOBAL][v]) ? newPhi : lsArray[GLOBAL][v];
          }
          else if (phi0*lsArray[GLOBAL][v] >= 0.0) {
            lsArray[GLOBAL][v] = PetscAbsReal(phi0) < PetscAbsReal(lsArray[GLOBAL][v]) ? phi0 : lsArray[GLOBAL][v];
          }
          else if (phi1*lsArray[GLOBAL][v] >= 0.0) {
            lsArray[GLOBAL][v] = PetscAbsReal(phi1) < PetscAbsReal(lsArray[GLOBAL][v]) ? phi1 : lsArray[GLOBAL][v];
          }
          else {
            throw std::runtime_error("Incorrect sign?\n");
          }
        }

      }
    }

    for (PetscInt v = 0; v < nTotalVert; ++v) {
      if (vertMask[v]==currentLevel) updatedVertex[v]=1;
    }
  }

  VecRestoreArray(lsVec[GLOBAL], &lsArray[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;

SaveData(vertDM, lsVec[GLOBAL], nTotalVert, vertList, "FMM.txt", 1);

  xexit("");

  DMRestoreWorkArray(vertDM, nTotalVert, MPIU_INT, &updatedVertex) >> ablate::utilities::PetscUtilities::checkError;


}


void Reconstruction::ToLevelSet(DM vofDM, Vec vofVec, const ablate::domain::Field vofField) {

int rank, size;
MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
MPI_Comm_size(PETSC_COMM_WORLD, &size);

DMViewFromOptions(vofDM, NULL, "-dm_view");
Reconstruction_SaveDM(vofDM, "mesh.txt");

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


  const PetscInt  dim = subDomain->GetDimensions();   // VOF and LS subdomains must have the same dimension. Can't think of a reason they wouldn't.
  PetscReal *closestPoint;
  DMGetWorkArray(vertGradDM, nLocalVert*dim, MPIU_REAL, &closestPoint) >> ablate::utilities::PetscUtilities::checkError;
  PetscInt *cpCell;
  DMGetWorkArray(vertDM, nLocalVert, MPIU_INT, &cpCell) >> ablate::utilities::PetscUtilities::checkError;

  InitalizeLevelSet(smoothVOFVec[LOCAL], cellMask, vertMask, lsVec, closestPoint, cpCell);
SaveData(vertDM, closestPoint, nLocalVert, vertList, "cp.txt", dim);

SaveData(vertDM, lsVec[LOCAL], nTotalVert, vertList, "vertLS0_L.txt", 1);
SaveData(vertDM, lsVec[GLOBAL], nLocalVert, vertList, "vertLS0_G.txt", 1);

FMM(cellMask, vertMask, lsVec);


xexit("");

  ReinitializeLevelSet(cellMask, vertMask, lsVec);

SaveData(vertDM, lsVec[LOCAL], nTotalVert, vertList, "vertLS1_L.txt", 1);
SaveData(vertDM, lsVec[GLOBAL], nLocalVert, vertList, "vertLS1_G.txt", 1);

xexit("");
  Vec curv[2];
  DMGetLocalVector(vertDM, &curv[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
  DMGetGlobalVector(vertDM, &curv[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;
  CalculateVertexCurvatures(cellMask, vertMask, lsVec, closestPoint, cpCell, curv);
SaveData(vertDM, curv[LOCAL], nLocalVert, vertList, "curv0.txt", 1);

//  Smooth(cellMask, vertMask, lsVec, curv);
SaveData(vertDM, curv[LOCAL], nLocalVert, vertList, "curv1.txt", 1);


  Extension(cellMask, vertMask, lsVec, closestPoint, cpCell, curv);
SaveData(vertDM, curv[LOCAL], nLocalVert, vertList, "curv2.txt", 1);



  DMRestoreLocalVector(cellDM, &curv[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
  DMRestoreGlobalVector(cellDM, &curv[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;
  DMRestoreWorkArray(vertDM, nTotalVert, MPIU_INT, &vertMask) >> ablate::utilities::PetscUtilities::checkError;




//  DMRestoreLocalVector(cellGradDM, &cellGradVec[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
//  DMRestoreGlobalVector(cellGradDM, &cellGradVec[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;

  DMRestoreLocalVector(cellDM, &smoothVOFVec[LOCAL]) >> utilities::PetscUtilities::checkError;
  DMRestoreGlobalVector(cellDM, &smoothVOFVec[GLOBAL]) >> utilities::PetscUtilities::checkError;

  if (subpointIndices) ISRestoreIndices(subpointIS, &subpointIndices) >> utilities::PetscUtilities::checkError;
xexit("");
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


