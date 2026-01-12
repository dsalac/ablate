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
  __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__); MPI_Barrier(PETSC_COMM_WORLD); exit(0);}

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
//  subDomain->GetRange(nullptr, 0, vertRange);
//  subDomain->GetCellRange(region, cellRange);   // Range of cells without boundary ghosts

  // Get the point->index mapping for cells
//  reverseVertRange = ablate::domain::ReverseRange(vertRange);
//  reverseCellRange = ablate::domain::ReverseRange(cellRange);


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

  // Get the start of any boundary ghost cells
  PetscInt boundaryCellStart;
  DMPlexGetCellTypeStratum(cellDM, DM_POLYTOPE_FV_GHOST, &boundaryCellStart, nullptr) >> utilities::PetscUtilities::checkError;
  boundaryCellStart = (boundaryCellStart > 0) ? boundaryCellStart : cEnd; // If there are no boundary cells then just use the last cell

  PetscMalloc2(cEnd - cStart, &cellList, cEnd - cStart, &reverseCellList) >> ablate::utilities::PetscUtilities::checkError;
  reverseCellList -= cStart;
  for (PetscInt c = 0; c < cEnd - cStart; ++c) cellList[c] = -1;

  for (PetscInt c = cStart; c < cEnd; ++c) {
    reverseCellList[c] = c - cStart;
  }

  // First the local cells
  nLocalCell = 0;
  for (PetscInt c = cStart; c < boundaryCellStart; ++c) {

    // See if it's owned by this rank
    PetscInt owned;
    DMPlexGetPointGlobal(cellDM, c, &owned, nullptr) >> utilities::PetscUtilities::checkError;

    if (owned >= 0) {
      cellList[nLocalCell++] = c;
    }
  }

  // Now the ghost cells. This should be unnecessary as the overlap cells should be nLocalCell -> boundaryCellStart-1
  nTotalCell = nLocalCell;
  for (PetscInt c = nLocalCell; c < boundaryCellStart; ++c) {

    // See if it's owned by this rank
    PetscInt owned;
    DMPlexGetPointGlobal(cellDM, c, &owned, nullptr) >> utilities::PetscUtilities::checkError;

    if (owned < 0) {
      cellList[nTotalCell++] = c;
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
  //BuildInterpGaussianList();

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

  MPI_Comm comm = PetscObjectComm((PetscObject)dm);
  int rank, size;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  PetscInt dim;

  DMGetDimension(dm, &dim);

  PetscInt eStart, eEnd;
  DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd) >> ablate::utilities::PetscUtilities::checkError;
  for (PetscInt r = 0; r < size; ++r) {
    if ( rank==r ) {

      FILE *f1;
      if ( rank==0 ) f1 = fopen(fname, "w");
      else f1 = fopen(fname, "a");
      if (f1==NULL) throw std::runtime_error("Vertex is marked as next to a cut cell but is not!");

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
    MPI_Barrier(comm);
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
      if (f1==NULL) throw std::runtime_error("Vertex is marked as next to a cut cell but is not!");

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

    MPI_Barrier(comm);
  }


  VecRestoreArrayRead(vec, &array) >> ablate::utilities::PetscUtilities::checkError;
  ablate::domain::RestoreRange(range);
}

void Reconstruction_SaveCellData(DM dm, const Vec vec, const char fname[255], const ablate::domain::Field *field, PetscInt Nc, std::shared_ptr<ablate::domain::SubDomain> subDomain) {
  Reconstruction_SaveCellData(dm, vec, fname, field->id, Nc, subDomain);
}

void Reconstruction::SaveData(DM dm, const PetscInt *array, const PetscInt nList, const PetscInt *list, const char fname[255], PetscInt Nc) {

  MPI_Comm      comm = PetscObjectComm((PetscObject)dm);
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  PetscInt      dim = subDomain->GetDimensions();

  for (PetscInt r = 0; r < size; ++r) {
    if ( rank==r ) {

      FILE *f1;
      if ( rank==0 ) f1 = fopen(fname, "w");
      else f1 = fopen(fname, "a");
      if (f1==NULL) throw std::runtime_error("Vertex is marked as next to a cut cell but is not!");

      for (PetscInt p = 0; p < nList; ++p) {
        const PetscInt point = list[p];
        PetscReal x[3];
        DMPlexComputeCellGeometryFVM(dm, point, NULL, x, NULL) >> ablate::utilities::PetscUtilities::checkError;

        for (PetscInt d = 0; d < dim; ++d) fprintf(f1, "%.16e\t", x[d]);

        for (PetscInt d = 0; d < Nc; ++d) fprintf(f1, "%" PetscInt_FMT"\t", array[p*Nc + d]);

        fprintf(f1, "\n");
      }
      fclose(f1);
    }

    MPI_Barrier(comm);
  }
}

void Reconstruction::SaveData(DM dm, const PetscScalar *array, const PetscInt nList, const PetscInt *list, const char fname[255], PetscInt Nc) {

  MPI_Comm      comm = PetscObjectComm((PetscObject)dm);
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  PetscInt      dim = subDomain->GetDimensions();

  for (PetscInt r = 0; r < size; ++r) {
    if ( rank==r ) {

      FILE *f1;
      if ( rank==0 ) f1 = fopen(fname, "w");
      else f1 = fopen(fname, "a");
      if (f1==NULL) throw std::runtime_error("Vertex is marked as next to a cut cell but is not!");

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

    MPI_Barrier(comm);
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
void Reconstruction::SetMasks(DM vofDM, Vec vofVec, const ablate::domain::Field vofField, const PetscInt nLevels, PetscInt *cellMask, PetscInt *vertMask) {

  int rank, size;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  MPI_Comm_size(PETSC_COMM_WORLD, &size);

  //PetscScalar *vofArray = nullptr; // This line is commented to have arbitrary interface
  PetscScalar *cellMaskVecArray = nullptr;
  Vec cellMaskVec[2] = {nullptr, nullptr};

  DMGetLocalVector(cellDM, &cellMaskVec[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
  DMGetGlobalVector(cellDM, &cellMaskVec[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;

  //VecGetArray(vofVec, &vofArray) >> ablate::utilities::PetscUtilities::checkError; // This line is commented to have arbitrary interface
  VecGetArray(cellMaskVec[LOCAL], &cellMaskVecArray);

  PetscBool CutCell = PETSC_FALSE;
  for (PetscInt c = 0; c < nLocalCell; ++c) {
    CutCell = Reconstruction::CutCellfromLS(vofDM, cellList[c], &vofField, vofVec); // Although we are using vofDM name here, but for arbitrary interface, the input is aux_dm
    cellMaskVecArray[c] = CutCell == PETSC_TRUE;
    //const PetscScalar *vof = nullptr; // This line is commented to have arbitrary interface
    //xDMPlexPointLocalRead(vofDM, cellList[c], vofField.id, vofArray, &vof) >> ablate::utilities::PetscUtilities::checkError; // This line is commented to have arbitrary interface
    //cellMaskVecArray[c] = cellMask[c] = ((*vof > 0.001) && (*vof < 0.999)); // This line is commented to have arbitrary interface
  }

  DMLocalToGlobal(cellDM, cellMaskVec[LOCAL], INSERT_VALUES, cellMaskVec[GLOBAL]);
  DMGlobalToLocal(cellDM, cellMaskVec[GLOBAL], INSERT_VALUES, cellMaskVec[LOCAL]);

  for (PetscInt c = 0; c < nTotalCell; ++c) cellMask[c] = cellMaskVecArray[c];

  //VecRestoreArray(vofVec, &vofArray) >> ablate::utilities::PetscUtilities::checkError; // This line is commented to have arbitrary interface
  VecRestoreArray(cellMaskVec[LOCAL], &cellMaskVecArray) >> ablate::utilities::PetscUtilities::checkError;

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

  // Now label the surrounding cells
  for (PetscInt l = 1; l <= nLevels; ++l) {

    VecZeroEntries(cellMaskVec[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
    VecGetArray(cellMaskVec[LOCAL], &cellMaskVecArray) >> ablate::utilities::PetscUtilities::checkError;

    for (PetscInt c = 0; c < nTotalCell; ++c) {
      if ( cellMask[c] == l ) {
        PetscInt cell = cellList[c];
        PetscInt nCells, *cells;
        DMPlexGetNeighbors(cellDM, cell, 1, -1.0, -1, PETSC_TRUE, PETSC_FALSE, &nCells, &cells) >> ablate::utilities::PetscUtilities::checkError;
        for (PetscInt i = 0; i < nCells; ++i) {
          const PetscInt id = reverseCellList[cells[i]];
          ++cellMaskVecArray[id];
        }
        DMPlexRestoreNeighbors(cellDM, cell, 1, -1.0, -1, PETSC_TRUE, PETSC_FALSE, &nCells, &cells) >> ablate::utilities::PetscUtilities::checkError;
      }
    }

    VecZeroEntries(cellMaskVec[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;
    DMLocalToGlobal(cellDM, cellMaskVec[LOCAL], ADD_VALUES, cellMaskVec[GLOBAL]) >> utilities::PetscUtilities::checkError;
    DMGlobalToLocal(cellDM, cellMaskVec[GLOBAL], INSERT_VALUES, cellMaskVec[LOCAL]) >> utilities::PetscUtilities::checkError;

    const PetscInt setValue = (l == nLevels) ? -1 : l + 1;
    for (PetscInt c = 0; c < nTotalCell; ++c) {
      if (cellMask[c] == 0 && cellMaskVecArray[c]>0.5) cellMask[c] = setValue;
    }
    VecRestoreArray(cellMaskVec[LOCAL], &cellMaskVecArray) >> ablate::utilities::PetscUtilities::checkError;
  }

  DMRestoreLocalVector(cellDM, &cellMaskVec[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
  DMRestoreGlobalVector(cellDM, &cellMaskVec[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;

  char filename[64];
  snprintf(filename, sizeof(filename), "cellmask_rank%d.txt", rank);
  FILE *file1 = fopen(filename, "w");
  const PetscInt dim = subDomain->GetDimensions();
  PetscReal x[dim]; // coordinates of cells
  for (PetscInt c = 0; c < nLocalCell; ++c) {
    DMPlexComputeCellGeometryFVM(vofDM, cellList[c], NULL, x, NULL) >> utilities::PetscUtilities::checkError; // Get the coordinates of a vertex
    PetscFPrintf(PETSC_COMM_SELF, file1, "%" PetscInt_FMT", %f, %f, %" PetscInt_FMT"\n", cellList[c], x[0], x[1], cellMask[c]);
  }
  fclose(file1);

  // Set the vertex mask-----------------------------------------------------------------------------------------------------------
  PetscScalar *vertMaskVecArray = nullptr;
  Vec vertMaskVec[2] = {nullptr, nullptr};

  DMGetLocalVector(vertDM, &vertMaskVec[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
  DMGetGlobalVector(vertDM, &vertMaskVec[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;

  VecGetArray(vertMaskVec[LOCAL], &vertMaskVecArray);

  for (PetscInt v = 0; v < nTotalVert; ++v) vertMaskVecArray[v] = nLevels + 2;

  if (rank==1) PetscPrintf(PETSC_COMM_SELF, "%d, %f\n", 5941, vertMaskVecArray[5941]);

  // First find the min level for each vertex considering neighboring cell levels
  for (PetscInt v = 0; v < nLocalVert; ++v) {
      PetscInt vertex = vertList[v];

      PetscInt nCell, *cells;
      DMPlexVertexGetCells(cellDM, vertex, &nCell, &cells) >> ablate::utilities::PetscUtilities::checkError;
      for (PetscInt c = 0; c < nCell; ++c) {
        const PetscInt id = reverseCellList[cells[c]];
        if (rank==1 && v == 5941) PetscPrintf(PETSC_COMM_SELF, "%d, %f, %" PetscInt_FMT", %" PetscInt_FMT"\n", 5941, vertMaskVecArray[5941], id, cellMask[id]);
        if (cellMask[id] == 0 || cellMask[id] == -1 || id >= nTotalCell) continue;
        vertMaskVecArray[v] = PetscMin(vertMaskVecArray[v], PetscAbsReal(cellMask[id]));
        if (rank==1 && v == 5941) PetscPrintf(PETSC_COMM_SELF, "%d, %f, %" PetscInt_FMT", %" PetscInt_FMT"\n", 5941, vertMaskVecArray[5941], id, cellMask[id]);
      }
      DMPlexCellRestoreVertices(cellDM, vertex, &nCell, &cells) >> ablate::utilities::PetscUtilities::checkError;
  }

  // Next set the additional vertices associated with boundary cells
  for (PetscInt c = 0; c < nTotalCell; ++c) {
    if (cellMask[c] == -1) {
      const PetscInt cell = cellList[c];

      PetscInt nVert, *verts;
      DMPlexCellGetVertices(vertDM, cell, &nVert, &verts) >> ablate::utilities::PetscUtilities::checkError;
      for (PetscInt v = 0; v < nVert; ++v) {
        const PetscInt id = reverseVertList[verts[v]];
        vertMaskVecArray[id] = (vertMaskVecArray[id]==(nLevels+2)) ? -1 : vertMaskVecArray[id];
      }
      DMPlexCellRestoreVertices(vertDM, cell, &nVert, &verts) >> ablate::utilities::PetscUtilities::checkError;
    }
  }

  // Switch all deactivated vertices to 0
  for (PetscInt v = 0; v < nTotalVert; ++v) {
    vertMaskVecArray[v] = (vertMaskVecArray[v]==(nLevels+2)) ? 0 : vertMaskVecArray[v];
  }

  DMLocalToGlobal(vertDM, vertMaskVec[LOCAL], INSERT_VALUES, vertMaskVec[GLOBAL]);
  DMGlobalToLocal(vertDM, vertMaskVec[GLOBAL], INSERT_VALUES, vertMaskVec[LOCAL]);

  for (PetscInt v = 0; v < nTotalVert; ++v) vertMask[v] = vertMaskVecArray[v];

  VecRestoreArray(vertMaskVec[LOCAL], &vertMaskVecArray) >> ablate::utilities::PetscUtilities::checkError;
  DMRestoreLocalVector(vertDM, &vertMaskVec[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
  DMRestoreGlobalVector(vertDM, &vertMaskVec[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;

  snprintf(filename, sizeof(filename), "vertmask_rank%d.txt", rank);
  FILE *file2 = fopen(filename, "w");
  for (PetscInt v = 0; v < nLocalVert; ++v) {
    DMPlexComputeCellGeometryFVM(vofDM, vertList[v], NULL, x, NULL) >> utilities::PetscUtilities::checkError; // Get the coordinates of a vertex
    PetscFPrintf(PETSC_COMM_SELF, file2, "%" PetscInt_FMT", %f, %f, %" PetscInt_FMT"\n", vertList[v], x[0], x[1], vertMask[v]);
  }
  fclose(file2);

}

// vofVec MUST have ghost cell information
void Reconstruction::InitalizeLevelSet(DM vofDM, Vec vofVec, const ablate::domain::Field vofField, const PetscInt *cellMask, const PetscInt *vertMask, Vec lsVec[2], PetscReal *closestPoint, PetscInt *cpCell) {

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
        printf("%" PetscInt_FMT";plot(%f,%f,'r*');\n", v, x[0], x[1]);
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
      DMPlexCellGradFromCell(vofDM, cellList[c], vofVec, vofField.id, 0, &cellGrad[c*dim]) >> ablate::utilities::PetscUtilities::checkError;
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
        const PetscScalar *vof;
        xDMPlexPointLocalRead(vofDM, cell, vofField.id, vofArray, &vof) >> ablate::utilities::PetscUtilities::checkError;

        PetscInt nv, *verts;
        DMPlexCellGetVertices(vertDM, cell, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;

        PetscReal *lsVertVals = NULL;
        DMGetWorkArray(vertDM, nv, MPIU_REAL, &lsVertVals) >> ablate::utilities::PetscUtilities::checkError;

        // Level set values at the vertices
        ablate::levelSet::Utilities::VertexLevelSet_VOF(vertDM, cell, *vof, &cellGrad[c*dim], &lsVertVals);

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

  // Get the closest point to each vertex, assuming a linear interface in a cell.
  PetscReal *cellPhi;
  DMGetWorkArray(cellDM, nTotalCell, MPIU_REAL, &cellPhi) >> ablate::utilities::PetscUtilities::checkError;

  PetscArrayzero(cellPhi, nTotalCell) >> ablate::utilities::PetscUtilities::checkError;
  const PetscScalar *lsArray = nullptr;
  VecGetArrayRead(lsVec[LOCAL], &lsArray) >> ablate::utilities::PetscUtilities::checkError;
  for (PetscInt c = 0; c < nTotalCell; ++c) {
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
  DMRestoreWorkArray(cellDM, nTotalCell, MPIU_REAL, &cellPhi) >> ablate::utilities::PetscUtilities::checkError;


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

    const PetscScalar *vof;
    xDMPlexPointLocalRead(vofDM, cell, vofField.id, vofArray, &vof) >> ablate::utilities::PetscUtilities::checkError;

    const PetscReal lsSetValues[2] = {lsRange[ *vof < 0.5 ? 1 : 0 ], PetscSignReal(0.5 - (*vof))*maxDist};
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
      }
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
}

PetscInt SolveQuadFormula(const PetscInt dim, PetscReal a[], PetscReal b[], PetscReal *phi) {

  PetscReal newPhi = *phi, phiIn = *phi;

//  // If there is no influence of a direction on the solution zero out that portion of the vector
//  for (PetscInt d = 0; d < dim; ++d) {
//    b[d] = (PetscAbsReal(a[d]) < PETSC_SMALL) ? 0.0 : b[d];
//  }

  const PetscReal p2 = ablate::utilities::MathUtilities::DotVector(dim, a, a);
  const PetscReal p1 = 2.0*ablate::utilities::MathUtilities::DotVector(dim, a, b);
  const PetscReal p0 = ablate::utilities::MathUtilities::DotVector(dim, b, b) - 1.0;

  PetscReal disc = p1*p1 - 4.0*p0*p2;
  //PetscPrintf(PETSC_COMM_SELF, "disc is %f\n", disc);

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

    if (phi0*phiIn >= 0.0 && phi1*phiIn >= 0.0) { // Both have the correct sign
      newPhi = PetscAbsReal(phi0) > PetscAbsReal(phi1) ? phi0 : phi1;
    }
    else if (phi0*phiIn >= 0.0) {
      newPhi = phi0;
    }
    else if (phi1*phiIn >= 0.0) {
      newPhi = phi1;
    }
    else {
      //throw std::runtime_error("Incorrect sign?\n");
      return 0;
    }

    if (PetscAbsReal(newPhi) < PetscAbsReal(phiIn)) {
      *phi = newPhi;
      return 1;
    }
    else {
      return 0;
    }
  }

  return 0;
}

void Reconstruction::FMM_CellBased(const PetscInt currentLevel, const PetscInt *cellMask, const PetscInt *vertMask, Vec updatedVec[2], Vec lsVec[2], Vec lsVecCopy[2], FILE *animFile) {

  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  PetscInt vStart, vEnd;
  DMPlexGetDepthStratum(vertDM, 0, &vStart, &vEnd);   // Range of vertices

  PetscInt dim;
  DMGetDimension(vertDM, &dim);

  MPI_Comm vertComm = PetscObjectComm((PetscObject)(vertDM));

  while (true) {
    // All work must be done on the local vector as a vertex associated with a cell might not be owned by this rank
    //  even if the cell is owned by this rank.
    PetscScalar *lsArray[2] = {nullptr, nullptr};
    VecGetArray(lsVec[LOCAL], &lsArray[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
    VecGetArray(lsVec[GLOBAL], &lsArray[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;

    PetscScalar *updatedVertex[2] = {nullptr, nullptr};
    VecGetArray(updatedVec[LOCAL], &updatedVertex[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
    VecGetArray(updatedVec[GLOBAL], &updatedVertex[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;

    PetscScalar *lstrue[2] = {nullptr, nullptr};
    VecGetArray(lsVecCopy[GLOBAL], &lstrue[GLOBAL]) >> utilities::PetscUtilities::checkError;
    VecGetArray(lsVecCopy[LOCAL], &lstrue[LOCAL]) >> utilities::PetscUtilities::checkError;

    PetscInt numVertUpdated = 0;
    for (PetscInt c = 0; c < nTotalCell; ++c) {
      if (cellMask[c]==currentLevel) {
        PetscInt cell = cellList[c];
        PetscInt nVert, *verts;
        DMPlexCellGetVertices(vertDM, cell, &nVert, &verts) >> ablate::utilities::PetscUtilities::checkError;

        PetscInt nSetVertices = 0, vertID = -1;
        for (PetscInt v = 0; v < nVert; ++v) {
        PetscInt id = reverseVertList[verts[v]];
          if (updatedVertex[LOCAL][id] > 0.5 && updatedVertex[LOCAL][id] < 1.5) {
            ++nSetVertices;
          }
          else {
            vertID = verts[v]; // When nUpdated+1 == nVert this will contain the ID of the single vertex to be updated
          }
        }

        const PetscInt id = reverseVertList[vertID];
        if (id < nLocalVert && nSetVertices + 1 == nVert) {
          // This will create the gradient vector a*phi + b where phi is the level set value to find
          PetscReal a[3] = {0.0, 0.0, 0.0}, b[3] = {0.0, 0.0, 0.0};

          PetscInt nFace;
          const PetscInt *faces;

          // Get all faces associated with the cell
          DMPlexGetConeSize(vertDM, cell, &nFace) >> ablate::utilities::PetscUtilities::checkError;
          DMPlexGetCone(vertDM, cell, &faces) >> ablate::utilities::PetscUtilities::checkError;
          for (PetscInt f = 0; f < nFace; ++f) {
            PetscReal N[3] = {0.0, 0.0, 0.0};
            DMPlexFaceCentroidOutwardAreaNormal(vertDM, cell, faces[f], NULL, N);

            // All points associated with this face
            PetscInt nClosure, *closure = NULL;
            DMPlexGetTransitiveClosure(vertDM, faces[f], PETSC_TRUE, &nClosure, &closure) >> ablate::utilities::PetscUtilities::checkError;

            PetscReal cnt = 0.0, ave = 0.0, vertCoeff = 0.0;
            for (PetscInt cl = 0; cl < nClosure * 2; cl += 2) {
              if (closure[cl] >= vStart && closure[cl] < vEnd) {  // Only use the points corresponding to a vertex

                const PetscInt clID = reverseVertList[closure[cl]];
                if (closure[cl]==vertID) {
                  if (updatedVertex[LOCAL][clID] > 0.5 && updatedVertex[LOCAL][clID] < 1.5) throw std::runtime_error("How can this be possible?\n");
                  ++vertCoeff;
                }
                else {
                  if (updatedVertex[LOCAL][clID] < 0.5 || updatedVertex[LOCAL][clID] > 1.5) throw std::runtime_error("How can this be possible?\n");
                  ave += lsArray[LOCAL][clID];

                }

                cnt += 1.0;
              }
            }

            DMPlexRestoreTransitiveClosure(vertDM, faces[f], PETSC_TRUE, &nClosure, &closure) >> ablate::utilities::PetscUtilities::checkError;

            // Function value at the face center
            ave /= cnt;
            vertCoeff /= cnt;
            for (PetscInt d = 0; d < dim; ++d) {
              a[d] += vertCoeff * N[d];
              b[d] += ave * N[d];
            }
          }

          PetscReal vol;
          DMPlexComputeCellGeometryFVM(vertDM, cell, &vol, NULL, NULL) >> ablate::utilities::PetscUtilities::checkError;
          for (PetscInt d = 0; d < dim; ++d) {
            a[d] /= vol;
            b[d] /= vol;
          }

          PetscInt result = SolveQuadFormula(dim, a, b, &lsArray[GLOBAL][id]);
          updatedVertex[GLOBAL][id] = PetscMin(updatedVertex[GLOBAL][id] + result, 1.0);
          numVertUpdated += (updatedVertex[GLOBAL][id] > 0.5);

          //PetscScalar Error =  PetscAbsScalar(lsArray[GLOBAL][id] - lstrue[GLOBAL][id]);
          //if (Error > 1e-3) {
            //PetscPrintf(PETSC_COMM_SELF, "level %" PetscInt_FMT", At cellbased, we have error %f\n", currentLevel, Error);
          //}

          updatedVertex[GLOBAL][id] = PetscMin(updatedVertex[GLOBAL][id] + result, 1.0);

          // Animation part
          if (updatedVertex[GLOBAL][id] > 0.5) {
              PetscReal x[dim];
              DMPlexComputeCellGeometryFVM(vertDM, vertList[id], NULL, x, NULL) >> ablate::utilities::PetscUtilities::checkError;
              fprintf(animFile, "%d %" PetscInt_FMT" %" PetscInt_FMT" %f %f %s\n", rank, currentLevel, id, x[0], x[1], "cellbased");
          }

        }
        DMPlexCellRestoreVertices(vertDM, cell, &nVert, &verts) >> ablate::utilities::PetscUtilities::checkError;
      }
    }

    VecRestoreArray(lsVec[LOCAL], &lsArray[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
    VecRestoreArray(lsVec[GLOBAL], &lsArray[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;
    DMGlobalToLocal(vertDM, lsVec[GLOBAL], INSERT_VALUES, lsVec[LOCAL]) >> utilities::PetscUtilities::checkError;

    VecRestoreArray(updatedVec[LOCAL], &updatedVertex[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
    VecRestoreArray(updatedVec[GLOBAL], &updatedVertex[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;
    DMGlobalToLocal(vertDM, updatedVec[GLOBAL], INSERT_VALUES, updatedVec[LOCAL]) >> utilities::PetscUtilities::checkError;

    VecRestoreArray(lsVecCopy[LOCAL], &lstrue[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
    VecRestoreArray(lsVecCopy[GLOBAL], &lstrue[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;

    MPI_Allreduce(MPI_IN_PLACE, &numVertUpdated, 1, MPIU_INT, MPIU_SUM, vertComm);
    if (numVertUpdated==0) break;
  }
}

void Reconstruction::FMM_CellBased_V3(const PetscInt currentLevel, const PetscInt *cellMask, const PetscInt *vertMask, Vec updatedVec[2], Vec lsVec[2]) {

  PetscInt vStart, vEnd;
  DMPlexGetDepthStratum(vertDM, 0, &vStart, &vEnd);   // Range of vertices

  PetscInt dim;
  DMGetDimension(vertDM, &dim);

  MPI_Comm vertComm = PetscObjectComm((PetscObject)(vertDM));

  PetscReal *numCells;
  DMGetWorkArray(vertDM, nLocalVert, MPIU_REAL, &numCells) >> ablate::utilities::PetscUtilities::checkError;

  while (true) {

    // All work must be done on the local vector as a vertex associated with a cell might not be owned by this rank
    //  even if the cell is owned by this rank.
    PetscScalar *lsArray[2] = {nullptr, nullptr};
    VecGetArray(lsVec[LOCAL], &lsArray[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
    VecGetArray(lsVec[GLOBAL], &lsArray[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;

    PetscScalar *updatedVertex[2] = {nullptr, nullptr};
    VecGetArray(updatedVec[LOCAL], &updatedVertex[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
    VecGetArray(updatedVec[GLOBAL], &updatedVertex[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;


    PetscArrayzero(numCells, nLocalVert);
    for (PetscInt v = 0; v < nLocalVert; ++v) {
      if (vertMask[v] == currentLevel && updatedVertex[GLOBAL][v] < 0.5) {
        lsArray[GLOBAL][v] = 0.0;
      }
    }

    PetscInt numVertUpdated = 0;
    for (PetscInt c = 0; c < nTotalCell; ++c) {
      if (cellMask[c]==currentLevel) {
        PetscInt cell = cellList[c];
        PetscInt nVert, *verts;
        DMPlexCellGetVertices(vertDM, cell, &nVert, &verts) >> ablate::utilities::PetscUtilities::checkError;

        PetscInt nSetVertices = 0, vertToUpdate = -1;
        for (PetscInt v = 0; v < nVert; ++v) {
          PetscInt id = reverseVertList[verts[v]];
          if (updatedVertex[LOCAL][id] > 0.5 && updatedVertex[LOCAL][id] < 1.5) {
            ++nSetVertices;
          }
          else {
            vertToUpdate = verts[v]; // When nUpdated+1 == nVert this will contain the ID of the single vertex to be updated
          }
        }

        const PetscInt id = reverseVertList[vertToUpdate];
        if (id < nLocalVert && nSetVertices + 1 == nVert) {
          // This will create the gradient vector a*phi + b where phi is the level set value to find
          PetscReal a[3] = {0.0, 0.0, 0.0}, b[3] = {0.0, 0.0, 0.0};

          PetscInt nFace;
          const PetscInt *faces;

          // Get all faces associated with the cell
          DMPlexGetConeSize(vertDM, cell, &nFace) >> ablate::utilities::PetscUtilities::checkError;
          DMPlexGetCone(vertDM, cell, &faces) >> ablate::utilities::PetscUtilities::checkError;
          for (PetscInt f = 0; f < nFace; ++f) {
              PetscReal N[3] = {0.0, 0.0, 0.0};
              DMPlexFaceCentroidOutwardAreaNormal(vertDM, cell, faces[f], NULL, N);

              // All points associated with this face
              PetscInt nClosure, *closure = NULL;
              DMPlexGetTransitiveClosure(vertDM, faces[f], PETSC_TRUE, &nClosure, &closure) >> ablate::utilities::PetscUtilities::checkError;

              PetscReal cnt = 0.0, ave = 0.0, vertCoeff = 0.0;
              for (PetscInt cl = 0; cl < nClosure * 2; cl += 2) {
                if (closure[cl] >= vStart && closure[cl] < vEnd) {  // Only use the points corresponding to a vertex

                  const PetscInt clID = reverseVertList[closure[cl]];
                  if (closure[cl]==vertToUpdate) {
                    if (updatedVertex[LOCAL][clID] > 0.5 && updatedVertex[LOCAL][clID] < 1.5) throw std::runtime_error("How can this be possible?\n");
                    ++vertCoeff;
                  }
                  else {
                    if (updatedVertex[LOCAL][clID] < 0.5 || updatedVertex[LOCAL][clID] > 1.5) throw std::runtime_error("How can this be possible?\n");
                    ave += lsArray[LOCAL][clID];
                  }

                  cnt += 1.0;
                }
              }

              DMPlexRestoreTransitiveClosure(vertDM, faces[f], PETSC_TRUE, &nClosure, &closure) >> ablate::utilities::PetscUtilities::checkError;

              // Function value at the face center
              ave /= cnt;
              vertCoeff /= cnt;
              for (PetscInt d = 0; d < dim; ++d) {
                  a[d] += vertCoeff * N[d];
                  b[d] += ave * N[d];
              }
          }

          PetscReal vol;
          DMPlexComputeCellGeometryFVM(vertDM, cell, &vol, NULL, NULL) >> ablate::utilities::PetscUtilities::checkError;
          for (PetscInt d = 0; d < dim; ++d) {
              a[d] /= vol;
              b[d] /= vol;
          }

          PetscReal newVal = lsArray[LOCAL][id];
          PetscInt result = SolveQuadFormula(dim, a, b, &newVal);

          if (result==1) {
            ++numCells[id];
            lsArray[GLOBAL][id] += 1.0/newVal;
          }



        }
        DMPlexCellRestoreVertices(vertDM, cell, &nVert, &verts) >> ablate::utilities::PetscUtilities::checkError;
      }
    }

    for (PetscInt v = 0; v < nLocalVert; ++v) {
      if (vertMask[v] == currentLevel && updatedVertex[GLOBAL][v] < 0.5) {
        if (numCells[v] > 0) {
          updatedVertex[GLOBAL][v] = 1.0;
          lsArray[GLOBAL][v] = numCells[v]/lsArray[GLOBAL][v];
          ++numVertUpdated;
        }
        else {
          lsArray[GLOBAL][v] = lsArray[LOCAL][v];
        }
      }

    }

    VecRestoreArray(lsVec[LOCAL], &lsArray[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
    VecRestoreArray(lsVec[GLOBAL], &lsArray[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;
    DMGlobalToLocal(vertDM, lsVec[GLOBAL], INSERT_VALUES, lsVec[LOCAL]) >> utilities::PetscUtilities::checkError;

    VecRestoreArray(updatedVec[LOCAL], &updatedVertex[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
    VecRestoreArray(updatedVec[GLOBAL], &updatedVertex[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;
    DMGlobalToLocal(vertDM, updatedVec[GLOBAL], INSERT_VALUES, updatedVec[LOCAL]) >> utilities::PetscUtilities::checkError;

    MPI_Allreduce(MPI_IN_PLACE, &numVertUpdated, 1, MPIU_INT, MPIU_SUM, vertComm);

    if (numVertUpdated==0) break;
  }

  DMRestoreWorkArray(vertDM, nLocalVert, MPIU_REAL, &numCells) >> ablate::utilities::PetscUtilities::checkError;

}

void Reconstruction::FMM_CellBased_V2(const PetscInt currentLevel, const PetscInt *cellMask, const PetscInt *vertMask, Vec updatedVec[2], Vec lsVec[2]) {

  PetscInt vStart, vEnd;
  DMPlexGetDepthStratum(vertDM, 0, &vStart, &vEnd);   // Range of vertices

  PetscInt dim;
  DMGetDimension(vertDM, &dim);

  MPI_Comm vertComm = PetscObjectComm((PetscObject)(vertDM));

  while (true) {

    // All work must be done on the local vector as a vertex associated with a cell might not be owned by this rank
    //  even if the cell is owned by this rank.
    PetscScalar *lsArray[2] = {nullptr, nullptr};
    VecGetArray(lsVec[LOCAL], &lsArray[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
    VecGetArray(lsVec[GLOBAL], &lsArray[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;

    PetscScalar *updatedVertex[2] = {nullptr, nullptr};
    VecGetArray(updatedVec[LOCAL], &updatedVertex[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
    VecGetArray(updatedVec[GLOBAL], &updatedVertex[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;

    PetscInt numVertUpdated = 0;

    for (PetscInt v = 0; v < nLocalVert; ++v) {

      if (vertMask[v]==currentLevel && updatedVertex[LOCAL][v] < 0.5) {

        PetscInt vert = vertList[v];

        PetscInt nCells, *cellList;
        DMPlexVertexGetCells(vertDM, vert, &nCells, &cellList) >> ablate::utilities::PetscUtilities::checkError;

        PetscInt validCells = 0;
        PetscReal a[3] = {0.0, 0.0, 0.0}, b[3] = {0.0, 0.0, 0.0};

        for (PetscInt c = 0; c < nCells; ++c) {
          PetscInt cell = cellList[c];

          PetscInt nVert, *vertList;
          DMPlexCellGetVertices(vertDM, cell, &nVert, &vertList) >> ablate::utilities::PetscUtilities::checkError;

          // Might want to create a list which tracks how many vertices have been updated for each cell to avoid doing this
          //  over and over again.
          PetscInt nUpdatedVerts = 0;
          for (PetscInt cv = 0; cv < nVert; ++cv) {
            PetscInt id = reverseVertList[vertList[cv]];
            if (updatedVertex[LOCAL][id] > 0.5) {
              ++nUpdatedVerts;
            }
          }

          if (nUpdatedVerts + 1 == nVert) {
            ++validCells;
            PetscReal cell_a[3] = {0.0, 0.0, 0.0}, cell_b[3] = {0.0, 0.0, 0.0};

            PetscInt nFace;
            const PetscInt *faces;

            // Get all faces associated with the cell
            DMPlexGetConeSize(vertDM, cell, &nFace) >> ablate::utilities::PetscUtilities::checkError;
            DMPlexGetCone(vertDM, cell, &faces) >> ablate::utilities::PetscUtilities::checkError;
            for (PetscInt f = 0; f < nFace; ++f) {
                PetscReal N[3] = {0.0, 0.0, 0.0};
                DMPlexFaceCentroidOutwardAreaNormal(vertDM, cell, faces[f], NULL, N);

                // All points associated with this face
                PetscInt nClosure, *closure = NULL;
                DMPlexGetTransitiveClosure(vertDM, faces[f], PETSC_TRUE, &nClosure, &closure) >> ablate::utilities::PetscUtilities::checkError;

                PetscReal cnt = 0.0, ave = 0.0, vertCoeff = 0.0;
                for (PetscInt cl = 0; cl < nClosure * 2; cl += 2) {
                  if (closure[cl] >= vStart && closure[cl] < vEnd) {  // Only use the points corresponding to a vertex

                    const PetscInt clID = reverseVertList[closure[cl]];
                    if (closure[cl]==vert) {
                      if (updatedVertex[LOCAL][clID] > 0.5 && updatedVertex[LOCAL][clID] < 1.5) throw std::runtime_error("How can this be possible?\n");
                      ++vertCoeff;
                    }
                    else {
                      if (updatedVertex[LOCAL][clID] < 0.5 || updatedVertex[LOCAL][clID] > 1.5) throw std::runtime_error("How can this be possible?\n");
                      ave += lsArray[LOCAL][clID];
                    }

                    cnt += 1.0;
                  }
                }

                DMPlexRestoreTransitiveClosure(vertDM, faces[f], PETSC_TRUE, &nClosure, &closure) >> ablate::utilities::PetscUtilities::checkError;

                // Function value at the face center
                ave /= cnt;
                vertCoeff /= cnt;
                for (PetscInt d = 0; d < dim; ++d) {
                    cell_a[d] += vertCoeff * N[d];
                    cell_b[d] += ave * N[d];
                }
              }

              PetscReal vol;
              DMPlexComputeCellGeometryFVM(vertDM, cell, &vol, NULL, NULL) >> ablate::utilities::PetscUtilities::checkError;
              for (PetscInt d = 0; d < dim; ++d) {
                  a[d] = cell_a[d]/vol;
                  b[d] = cell_b[d]/vol;
              }
            }

          DMPlexCellRestoreVertices(vertDM, cell, &nVert, &vertList) >> ablate::utilities::PetscUtilities::checkError;

        }

        DMPlexVertexRestoreCells(vertDM, vert, &nCells, &cellList) >> ablate::utilities::PetscUtilities::checkError;

        if (validCells > 0) {

printf("%" PetscInt_FMT"\n", validCells);
printf("%+f\t%+f\n", a[0], b[0]);
printf("%+f\t%+f\n", a[1], b[1]);

          PetscInt result = SolveQuadFormula(dim, a, b, &lsArray[GLOBAL][v]);
          updatedVertex[GLOBAL][v] = PetscMin(updatedVertex[GLOBAL][v] + result, 1.0);
          numVertUpdated += (updatedVertex[GLOBAL][v] > 0.5);

PetscReal x[dim];
DMPlexComputeCellGeometryFVM(vertDM, vert, NULL, x, NULL) >> ablate::utilities::PetscUtilities::checkError;
xexit("%+f\t%+f\n", lsArray[GLOBAL][v], 1 - sqrt(x[0]*x[0] + x[1]*x[1]));
        }
      }
    }


    VecRestoreArray(lsVec[LOCAL], &lsArray[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
    VecRestoreArray(lsVec[GLOBAL], &lsArray[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;
    DMGlobalToLocal(vertDM, lsVec[GLOBAL], INSERT_VALUES, lsVec[LOCAL]) >> utilities::PetscUtilities::checkError;

    VecRestoreArray(updatedVec[LOCAL], &updatedVertex[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
    VecRestoreArray(updatedVec[GLOBAL], &updatedVertex[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;
    DMGlobalToLocal(vertDM, updatedVec[GLOBAL], INSERT_VALUES, updatedVec[LOCAL]) >> utilities::PetscUtilities::checkError;

    MPI_Allreduce(MPI_IN_PLACE, &numVertUpdated, 1, MPIU_INT, MPIU_SUM, vertComm);

    if (numVertUpdated==0) break;
  }

}

PetscInt Reconstruction::FFM_VertexBased_Solve(const PetscInt dim, const PetscInt minVerts, const PetscReal x0[], const PetscInt nVert, PetscInt verts[], PetscScalar *updatedVertex, PetscScalar *lsArray, PetscReal *updatedLS, const PetscInt *vertMask, const PetscInt currentLevel) {

  PetscReal phiList[nVert-1];
  PetscInt vertOrder[nVert-1], vertIDs[nVert-1], nValid = 0;

  // Get the list of valid neighbors and their level set values
  for (PetscInt nv = 0; nv < nVert; ++nv) {
    const PetscInt neighbor = verts[nv];
    const PetscInt id = reverseVertList[neighbor]; // id is the index of vertex in vertlist. For example, vertlist[52] = 195. Here neighbor is 195 and id which came from reverseVertlist[neighbor] is the index which is 52.

    if (updatedVertex[id] > 0.5 && updatedVertex[id] < 1.5 && vertMask[id]==currentLevel-1) {
      phiList[nValid] = PetscAbsReal(lsArray[id]);
      vertOrder[nValid] = nValid;
      vertIDs[nValid++] = id;
    }
  }

  if (nValid < dim) return 0;

  // Calculate the permutation list to make phiList ordered
  PetscSortRealWithPermutation(nValid, phiList, vertOrder);

  char transpose = 'T';
  PetscBLASInt m, n = 0, nrhs = 2, info, worksize;
  PetscBLASIntCast(dim, &m) >> ablate::utilities::PetscUtilities::checkError;
  PetscBLASIntCast(2*nValid, &worksize) >> ablate::utilities::PetscUtilities::checkError;
  double work[worksize];
  PetscReal P[nValid*dim], a[nValid], b[nValid], rhs[2*nValid];

  // Set the data for the solve
  for (PetscInt i = 0; i < nValid; ++i) {
    // Order the data from largest level set to smallest level set
    PetscInt id = vertIDs[vertOrder[nValid-1-i]];

    // This doesn't seem to make a difference for simplex meshes
    // Order the data from smallest level set to largest level set
//    PetscInt id = vertIDs[vertOrder[i]];

    PetscReal x[dim];
    DMPlexComputeCellGeometryFVM(vertDM, vertList[id], NULL, x, NULL) >> ablate::utilities::PetscUtilities::checkError;

    for (PetscInt d = 0; d < dim; ++d) {
      P[n*dim + d] = x[d] - x0[d];
    }

    // It might not be necessary to do this, but any change in error needs to be investigated first
    PetscReal mag = ablate::utilities::MathUtilities::MagVector(dim, &P[n*dim]);
    ablate::utilities::MathUtilities::ScaleVector(dim, &P[n*dim], 1.0/mag);
    a[n] = -1.0/mag;
    b[n++] = lsArray[id]/mag;

  }

  // When the vertex is equal distance between two interfaces the gradient will be zero.
  //  In this case remove points one at a time until a non-zero gradient is calculated.
  while (n >= minVerts) {
    for (PetscInt i = 0; i < n; ++i) {
      rhs[i] = a[i];
      rhs[i + n] = b[i];
    }

    LAPACKgels_(&transpose, &m, &n, &nrhs, P, &m, rhs, &n, work, &worksize, &info);
    if (info != 0) throw std::runtime_error("Bad argument to GELS");

    PetscReal p2 = ablate::utilities::MathUtilities::DotVector(dim, &rhs[0], &rhs[0]);

    if (p2 > PETSC_SMALL) break;
    --n;
  }

  if (n < minVerts) return 0;

  return SolveQuadFormula(dim, &rhs[0], &rhs[n], updatedLS);
}

//void Reconstruction::FMM_VertexBased_V2(const PetscInt currentLevel, const PetscInt *cellMask, const PetscInt *vertMask, Vec updatedVec[2], Vec lsVec[2], Vec lsVecCopy[2]) {
  //PetscInt dim;
  //DMGetDimension(vertDM, &dim);

  //MPI_Comm vertComm = PetscObjectComm((PetscObject)(vertDM));

  //while (true) {

    //PetscScalar *lsArray[2] = {nullptr, nullptr};
    //VecGetArray(lsVec[LOCAL], &lsArray[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
    //VecGetArray(lsVec[GLOBAL], &lsArray[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;

    //PetscScalar *updatedVertex[2] = {nullptr, nullptr};
    //VecGetArray(updatedVec[LOCAL], &updatedVertex[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
    //VecGetArray(updatedVec[GLOBAL], &updatedVertex[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;

    //PetscScalar *lstrue[2] = {nullptr, nullptr};
  //VecGetArray(lsVecCopy[GLOBAL], &lstrue[GLOBAL]) >> utilities::PetscUtilities::checkError;
  //VecGetArray(lsVecCopy[LOCAL], &lstrue[LOCAL]) >> utilities::PetscUtilities::checkError;

    //PetscInt numVertUpdated = 0;
    //for (PetscInt v = 0; v < nLocalVert; ++v) {

      //if (vertMask[v]==currentLevel && updatedVertex[LOCAL][v] < 0.5) {

        //PetscInt vert = vertList[v];

        //PetscInt nCells, *cells;
        //DMPlexVertexGetCells(vertDM, vert, &nCells, &cells);

        //PetscReal x0[dim];
        //DMPlexComputeCellGeometryFVM(vertDM, vert, NULL, x0, NULL) >> ablate::utilities::PetscUtilities::checkError;

        //for (PetscInt c = 0; c < nCells; ++c) {
          //PetscInt cell = cells[c];

          //if (ablate::levelSet::Utilities::ValidCell(vertDM, cell)) {

            //PetscInt nVert, *cellVerts;
            //DMPlexCellGetVertices(vertDM, cell, &nVert, &cellVerts);

            //PetscInt result = FFM_VertexBased_Solve(dim, dim, x0, nVert, cellVerts, updatedVertex[LOCAL], lsArray[LOCAL], &lsArray[GLOBAL][v]);
            //updatedVertex[GLOBAL][v] = PetscMin(updatedVertex[GLOBAL][v] + result, 1.0);
            //numVertUpdated += (updatedVertex[GLOBAL][v] > 0.5);

            //DMPlexCellRestoreVertices(vertDM, cell, &nVert, &cellVerts);
          //}

        //}

        //DMPlexVertexRestoreCells(vertDM, vert, &nCells, &cells);

      //}
    //}
    //VecRestoreArray(lsVec[LOCAL], &lsArray[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
    //VecRestoreArray(lsVec[GLOBAL], &lsArray[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;
    //VecRestoreArray(updatedVec[LOCAL], &updatedVertex[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
    //VecRestoreArray(updatedVec[GLOBAL], &updatedVertex[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;
    //DMGlobalToLocal(vertDM, lsVec[GLOBAL], INSERT_VALUES, lsVec[LOCAL]) >> utilities::PetscUtilities::checkError;
    //DMGlobalToLocal(vertDM, updatedVec[GLOBAL], INSERT_VALUES, updatedVec[LOCAL]) >> utilities::PetscUtilities::checkError;

    //VecRestoreArray(lsVecCopy[LOCAL], &lstrue[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
    //VecRestoreArray(lsVecCopy[GLOBAL], &lstrue[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;

  //int rank;
  //MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  //printf("Vrt2 %" PetscInt_FMT" %" PetscInt_FMT": %" PetscInt_FMT"\n", rank, currentLevel, numVertUpdated);

    //MPI_Allreduce(MPI_IN_PLACE, &numVertUpdated, 1, MPIU_INT, MPIU_SUM, vertComm);
////xexit("%ld\n",  numVertUpdated);
    //if (numVertUpdated==0) break;

  //}

  //PetscScalar *updatedVertex[2] = {nullptr, nullptr};
  //VecGetArray(updatedVec[LOCAL], &updatedVertex[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
  //PetscInt nNOTSetVertices = 0;
  //for (PetscInt v = 0; v < nLocalVert; ++v) {
    //if (vertMask[v]==currentLevel && updatedVertex[LOCAL][v] < 0.5) {
      //++nNOTSetVertices;
    //}
  //}
  //printf("vertexbased_v2, number of unset vertices in level %" PetscInt_FMT" is %" PetscInt_FMT"\n", currentLevel, nNOTSetVertices);
  //VecRestoreArray(updatedVec[LOCAL], &updatedVertex[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;

//}

//void Reconstruction::FMM_VertexBased_V1(const PetscInt currentLevel, const PetscInt minVerts, const PetscInt *cellMask, const PetscInt *vertMask, Vec updatedVec[2], Vec lsVec[2], Vec lsVecCopy[2], FILE *animFile) {

  //int rank;
  //MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  //PetscInt dim;
  //DMGetDimension(vertDM, &dim);

  //MPI_Comm vertComm = PetscObjectComm((PetscObject)(vertDM));

  //while (true) {

    //PetscScalar *lsArray[2] = {nullptr, nullptr};
    //VecGetArray(lsVec[LOCAL], &lsArray[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
    //VecGetArray(lsVec[GLOBAL], &lsArray[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;

    //PetscScalar *updatedVertex[2] = {nullptr, nullptr};
    //VecGetArray(updatedVec[LOCAL], &updatedVertex[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
    //VecGetArray(updatedVec[GLOBAL], &updatedVertex[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;

    //PetscScalar *lstrue[2] = {nullptr, nullptr};
  //VecGetArray(lsVecCopy[GLOBAL], &lstrue[GLOBAL]) >> utilities::PetscUtilities::checkError;
  //VecGetArray(lsVecCopy[LOCAL], &lstrue[LOCAL]) >> utilities::PetscUtilities::checkError;

    //PetscInt numVertUpdated = 0;
    //for (PetscInt v = 0; v < nLocalVert; ++v) {

      //if (vertMask[v]==currentLevel && updatedVertex[LOCAL][v] < 0.5) {

        //PetscInt vert = vertList[v];

        //PetscReal x0[dim];
        //DMPlexComputeCellGeometryFVM(vertDM, vert, NULL, x0, NULL) >> ablate::utilities::PetscUtilities::checkError;

        //PetscInt nVert, *neighborVerts;
        //DMPlexGetNeighbors(vertDM, vert, 1, -1.0, -1, PETSC_TRUE, PETSC_TRUE, &nVert, &neighborVerts); // Return neighboring vertices of a vertex by one level and including the corner ones

        //PetscInt result = FFM_VertexBased_Solve(dim, minVerts, x0, nVert, neighborVerts, updatedVertex[LOCAL], lsArray[LOCAL], &lsArray[GLOBAL][v]);
        //updatedVertex[GLOBAL][v] = PetscMin(updatedVertex[GLOBAL][v] + result, 1.0);
        //numVertUpdated += (updatedVertex[GLOBAL][v] > 0.5);

        //PetscScalar Error =  PetscAbsScalar(lsArray[GLOBAL][v] - lstrue[GLOBAL][v]);
    //if (Error > 1e-3) {
      //PetscPrintf(PETSC_COMM_SELF, "level %" PetscInt_FMT", At vertexbased_v1, we have error %f\n", currentLevel, Error);
    //}

    //// Animation part
    //if (updatedVertex[GLOBAL][v] > 0.5) {
        //PetscReal x[dim];
        //DMPlexComputeCellGeometryFVM(vertDM, vertList[v], NULL, x, NULL) >> ablate::utilities::PetscUtilities::checkError;
        //fprintf(animFile, "%" PetscInt_FMT" %" PetscInt_FMT" %" PetscInt_FMT" %f %f %s\n", rank, currentLevel, v, x[0], x[1], "vertexbased_v1");
    //}

        //DMPlexRestoreNeighbors(vertDM, vert, 1, -1.0, -1, PETSC_TRUE, PETSC_TRUE, &nVert, &neighborVerts);
////if (currentLevel>11) {
////  char fname[255];
////  sprintf(fname, "mid%ld.txt", currentLevel);
////  SaveData(vertDM, lsVec[GLOBAL], nLocalVert, vertList, fname, 1);
////}
      //}
    //}
    //VecRestoreArray(lsVec[LOCAL], &lsArray[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
    //VecRestoreArray(lsVec[GLOBAL], &lsArray[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;
    //VecRestoreArray(updatedVec[LOCAL], &updatedVertex[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
    //VecRestoreArray(updatedVec[GLOBAL], &updatedVertex[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;
    //DMGlobalToLocal(vertDM, lsVec[GLOBAL], INSERT_VALUES, lsVec[LOCAL]) >> utilities::PetscUtilities::checkError;
    //DMGlobalToLocal(vertDM, updatedVec[GLOBAL], INSERT_VALUES, updatedVec[LOCAL]) >> utilities::PetscUtilities::checkError;

    //VecRestoreArray(lsVecCopy[LOCAL], &lstrue[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
    //VecRestoreArray(lsVecCopy[GLOBAL], &lstrue[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;

  //int rank;
  //MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  //printf("Vrt1 %" PetscInt_FMT" %" PetscInt_FMT": %" PetscInt_FMT"\n", rank, currentLevel, numVertUpdated);

    //MPI_Allreduce(MPI_IN_PLACE, &numVertUpdated, 1, MPIU_INT, MPIU_SUM, vertComm);

    //if (numVertUpdated==0) break;

  //}

  //PetscScalar *updatedVertex[2] = {nullptr, nullptr};
  //VecGetArray(updatedVec[LOCAL], &updatedVertex[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
  //PetscInt nNOTSetVertices = 0;
  //for (PetscInt v = 0; v < nLocalVert; ++v) {
    //if (vertMask[v]==currentLevel && updatedVertex[LOCAL][v] < 0.5) {
      //++nNOTSetVertices;
    //}
  //}
  //printf("vertexbased_v1, number of unset vertices in level %" PetscInt_FMT" is %" PetscInt_FMT"\n", currentLevel, nNOTSetVertices);
  //VecRestoreArray(updatedVec[LOCAL], &updatedVertex[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;

//}

// VertexBased_GeneralModifiedGreenGauss
PetscInt Reconstruction::FFM_VertexBased_GMGG(const PetscInt dim, PetscInt id_potential, PetscInt id_base, PetscScalar *updatedVertex, PetscScalar *xCoord, PetscScalar *yCoord, const PetscInt *vertMask, const PetscInt currentLevel, PetscScalar *lsArray, PetscReal *updatedLS) {

  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  if (rank==1) PetscPrintf(PETSC_COMM_SELF, "id_potential starts here %" PetscInt_FMT"\n", id_potential);
  // Do the calculations for each valid base vertex
  PetscReal best_phi = PETSC_MAX_REAL;
  PetscInt result = 0;
  for (PetscInt i=0; i<2;++i) {
    PetscInt level_limit;
    if (i==0) {
      level_limit = currentLevel-1;
    } else {
      level_limit = currentLevel;
    }
    DMSetBasicAdjacency(vertDM, PETSC_FALSE, PETSC_FALSE); // Connected vertices
    PetscInt nBase = PETSC_DETERMINE;
    PetscInt *baseVerts = NULL;
    DMPlexGetAdjacency(vertDM, vertList[id_potential], &nBase, &baseVerts); // just the points that are connected with edge
    PetscInt baseIDs[nBase], nValidBase = 0;
    for (PetscInt nv = 0; nv < nBase; ++nv) {
      PetscInt id = reverseVertList[baseVerts[nv]];
      if (updatedVertex[id] > 0.5 && updatedVertex[id] < 1.5 && vertMask[id] <= level_limit) {
      baseIDs[nValidBase++] = id;
      }
    }

    if (rank==1) {
    for (PetscInt v = 0; v < nValidBase; ++v) {
      PetscPrintf(PETSC_COMM_SELF, "baseid is %" PetscInt_FMT" and i is %" PetscInt_FMT"\n", baseIDs[v], v);
    }
    }

    PetscInt best_stencil_size = 0;
    for (PetscInt bv = 0; bv <  nValidBase; ++bv) {
      PetscInt PnCells, *Pcells;
      DMPlexVertexGetCells(vertDM, vertList[id_potential], &PnCells, &Pcells);
      id_base = baseIDs[bv];
      PetscInt BnCells, *Bcells;
      DMPlexVertexGetCells(vertDM, vertList[id_base], &BnCells, &Bcells);

      PetscInt cellIDs[8], nCells = 0; // shared cells of base vertex and potential vertex
      for (PetscInt i = 0; i < PnCells; ++i) {
        for (PetscInt j = 0; j < BnCells; ++j) {
          DMPolytopeType cellType;
          DMPlexGetCellType(vertDM, Pcells[i], &cellType);
          if (Pcells[i] == Bcells[j] && cellType != 12) {
            cellIDs[nCells++] = Pcells[i];
          }
        }
      }

      PetscInt stencilIDs[8], nStencil = 0;

      // helper lambda to check membership in baseIDs[]
        auto isInBaseList = [&](PetscInt id) {
        for (PetscInt v = 0; v < nValidBase; ++v) {
          if (baseIDs[v] == id) return true;
        }
        return false;
      };

      for (PetscInt i = 0; i < nCells; ++i) {
        PetscInt nCellVert, *cellverts;
        DMPlexCellGetVertices(vertDM, cellIDs[i], &nCellVert, &cellverts) >>
        ablate::utilities::PetscUtilities::checkError;
        for (PetscInt v = 0; v < nCellVert; ++v) {
          PetscInt id = reverseVertList[cellverts[v]];
          if (id != id_base && updatedVertex[id] > 0.5 && updatedVertex[id] < 1.5 && vertMask[id] <= level_limit && !isInBaseList(id)) {
            stencilIDs[nStencil++] = id;
          }
        }
      }

      if (rank==1) PetscPrintf(PETSC_COMM_SELF, "idbase %" PetscInt_FMT"\n", id_base);
      // always insert id_base
      stencilIDs[nStencil++] = id_base;
      if (rank==1) {
      for (PetscInt v = 0; v < nStencil; ++v) {
        PetscPrintf(PETSC_COMM_SELF, "idstencil is %" PetscInt_FMT" with x %.14f and y %.14f and ls is %.14f\n", stencilIDs[v], xCoord[stencilIDs[v]], yCoord[stencilIDs[v]], lsArray[stencilIDs[v]]);
      }
      }
      if (nStencil < dim) continue;

      PetscReal best_phi_local = PETSC_MAX_REAL;
      auto TrySubset = [&](PetscInt *inputstencils, PetscInt nStencil, PetscReal &best_phi_local, PetscReal &best_phi, PetscInt &result, PetscInt *mainstencils, PetscInt nmainStencil) {

        std::sort(inputstencils, inputstencils + nStencil, [&](PetscInt a, PetscInt b) {
          return PetscAbsReal(lsArray[a]) < PetscAbsReal(lsArray[b]);
        });

        if (rank==1) {
        for (PetscInt v = 0; v < nStencil; ++v) {
          PetscPrintf(PETSC_COMM_SELF, "idstencil with nStencil %" PetscInt_FMT" is %" PetscInt_FMT" with x %.14f and y %.14f and ls is %.14f\n", nStencil, inputstencils[v], xCoord[inputstencils[v]], yCoord[inputstencils[v]], lsArray[inputstencils[v]]);
        }
        }

          // Build edges
          struct Edge {
          PetscInt v1, v2;
        };
          PetscInt nallEdge = 0;
          Edge alledges[nStencil+1];

        for (PetscInt v = 0; v < nStencil; ++v) {
                if (inputstencils[v] != id_base) {
            alledges[nallEdge++] = { std::min(id_potential, inputstencils[v]), std::max(id_potential, inputstencils[v]) };
                    if (nStencil == 3) alledges[nallEdge++] = { std::min(id_base, inputstencils[v]), std::max(id_base, inputstencils[v]) };
                }
                if (inputstencils[v] == id_base && nStencil == dim) {
            alledges[nallEdge++] = { std::min(id_potential, inputstencils[v]), std::max(id_potential, inputstencils[v]) };
                }
            }
            if (nStencil == 2) alledges[nallEdge++] = { std::min(inputstencils[0], inputstencils[1]), std::max(inputstencils[0], inputstencils[1]) }; //works for 2D

            PetscReal a[3] = {0.0, 0.0, 0.0}, b[3] = {0.0, 0.0, 0.0};
            PetscScalar N[3];
            PetscScalar vol = 0.0;
            PetscInt edge[2];
        for (PetscInt i = 0; i < nallEdge; ++i) {
          edge[0]= alledges[i].v1;
          edge[1] = alledges[i].v2;
          PetscScalar vec[dim];
          for (int d = 0; d < 3; d++) N[d] = 0.0;
          vec[0] = xCoord[edge[1]] - xCoord[edge[0]];
          vec[1] = yCoord[edge[1]] - yCoord[edge[0]];
          N[0] = vec[1];
          N[1] = -vec[0];

          PetscScalar midvec[2];
          midvec[0] = 0.5*(xCoord[edge[1]] + xCoord[edge[0]]);
          midvec[1] = 0.5*(yCoord[edge[1]] + yCoord[edge[0]]);

          PetscScalar cellcenter[2] = {xCoord[id_potential], yCoord[id_potential] };
          for (PetscInt v = 0; v < nStencil; ++v) {
            cellcenter[0] += xCoord[inputstencils[v]];
            cellcenter[1] += yCoord[inputstencils[v]];
          }
          cellcenter[0] /= nStencil+1;
          cellcenter[1] /= nStencil+1;

          PetscScalar centertomid[dim];
          for (PetscInt d = 0; d < dim; ++d) {
            centertomid[d] = cellcenter[d] -  midvec[d];
          }

          if (N[0]*centertomid[0] + N[1]*centertomid[1] > 0.0) {
            N[0] = -N[0];
            N[1] = -N[1];
          }

          for (PetscInt d = 0; d < dim; ++d) {
            for (PetscInt k = 0; k < 2; ++k) {
              PetscInt v = edge[k];
              if (updatedVertex[v] > 0.5 && updatedVertex[v] < 1.5) {
                b[d] += 0.5 * lsArray[v] * N[d];
              } else {
                a[d] += 0.5 * N[d];
              }
            }
          }

          if (rank==1) PetscPrintf(PETSC_COMM_SELF,"%" PetscInt_FMT" and %" PetscInt_FMT"\n", edge[0], edge[1]);
          if (rank==1) PetscPrintf(PETSC_COMM_SELF,"%f and %f\n", xCoord[edge[0]], yCoord[edge[0]]);
          if (rank==1) PetscPrintf(PETSC_COMM_SELF,"%f and %f\n", N[0], N[1]);
            }

            PetscInt start = alledges[0].v1;  // first vertex of e0
            PetscInt prev = -1;
            PetscInt cur = start;

            PetscInt orderedVerts[nStencil+1]; // number of vertices = number of unique vertices
            PetscInt nVerts = 0;
            orderedVerts[nVerts++] = cur;  // first vertex

            while (nVerts < nStencil+1) { // number of unique vertices
                for (PetscInt i = 0; i < nStencil+1; ++i) {  // loop over all edges //nStencil+1
                    PetscInt v0 = alledges[i].v1;
                    PetscInt v1 = alledges[i].v2;

                    PetscInt next = -1;
                    if (v0 == cur && v1 != prev) next = v1;
                    else if (v1 == cur && v0 != prev) next = v0;

                    if (next != -1) {
                        orderedVerts[nVerts++] = next;
                        prev = cur;
                        cur = next;
                        break; // move to next vertex
                    }
                }

            }

            PetscScalar temp_area = 0.0;
            for (PetscInt i = 0; i < nVerts; ++i) {
                PetscInt j = (i + 1) % nVerts; // next vertex, wrap around
                PetscInt vi = orderedVerts[i];
                PetscInt vj = orderedVerts[j];
                temp_area += xCoord[vi] * yCoord[vj] - xCoord[vj] * yCoord[vi];
            }

            vol = 0.5 * PetscAbsReal(temp_area);
            if (rank==1) PetscPrintf(PETSC_COMM_SELF, "vol is %f\n", vol);

        for (PetscInt d = 0; d < dim; ++d) {
          if (rank==1) PetscPrintf(PETSC_COMM_SELF, "a is %f and b is %f\n", a[d], b[d]);
              a[d] /= vol;
              b[d] /= vol;
            }

            PetscBool accept = PETSC_FALSE;
        PetscReal temp_phi = *updatedLS;
            if (rank==1)PetscPrintf(PETSC_COMM_SELF,"Try with %" PetscInt_FMT" stencils -> φ = %f\n", nStencil, temp_phi);
            PetscInt temp_result = SolveQuadFormula(dim, a, b, &temp_phi);
            if (rank==1)PetscPrintf(PETSC_COMM_SELF, "Try with %" PetscInt_FMT" stencils -> φ = %f, result=%" PetscInt_FMT"\n", nStencil, temp_phi, result);
        if (rank==1)PetscPrintf(PETSC_COMM_SELF, "id_potential is %" PetscInt_FMT" and tempphi is %f and result is %" PetscInt_FMT"\n", id_potential, temp_phi, temp_result);
            if (temp_result != 1) return false; //no quadratic solution and return temp_result which is zero

            PetscBool monotone = PETSC_TRUE;
        for (PetscInt v = 0; v < nmainStencil; ++v) {
          if (PetscAbsReal(temp_phi) < PetscAbsReal(lsArray[mainstencils[v]])) {
            if (rank==1) PetscPrintf(PETSC_COMM_SELF, "Rejected φ=%f because neighbor %" PetscInt_FMT" has |φ| smaller.\n", temp_phi, mainstencils[v]);
                    monotone = PETSC_FALSE;
                    return false;
          }
        }

        if (monotone) {
          accept = PETSC_TRUE;
          best_phi_local = temp_phi;
        }

            //if (accept && PetscAbsReal(best_phi_local) < PetscAbsReal(best_phi)) {
              //best_phi = best_phi_local;
              //result = temp_result;
            //}
            if (accept) {
            if (nStencil > best_stencil_size || (nStencil == best_stencil_size && PetscAbsReal(temp_phi) < PetscAbsReal(best_phi))) {
                best_phi = temp_phi;
                best_stencil_size = nStencil;
                result = temp_result;
            }
        }
            return true;
      };

      // full stencil for a base
      TrySubset(stencilIDs, nStencil, best_phi_local, best_phi, result, stencilIDs, nStencil);

      // all pairs
      for (PetscInt i = 0; i < nStencil; ++i) {
          for (PetscInt j = i + 1; j < nStencil; ++j) {
              PetscInt pair[2] = { stencilIDs[i], stencilIDs[j] };
              TrySubset(pair, 2, best_phi_local, best_phi, result, stencilIDs, nStencil);
          }
      }

  ////PetscPrintf(PETSC_COMM_SELF, "bestphi is %f\n", best_phi);
  //for (PetscInt usedStencil = nStencil; usedStencil >= dim && !accept; --usedStencil) {
        ////std::sort(stencilIDs, stencilIDs + usedStencil, [&](PetscInt a, PetscInt b){
                    ////return PetscAbsReal(lsArray[a]) < PetscAbsReal(lsArray[b]);
              ////});
        //PetscInt stencilStart = nStencil - usedStencil;

        //struct Edge {
          //PetscInt v1, v2;
        //};
        //Edge alledges[usedStencil+1];
        //PetscInt nallEdge = 0;

        //for (PetscInt v = stencilStart; v < nStencil; ++v) { //for (PetscInt v = 0; v < usedStencil; ++v)
            //if (stencilIDs[v] != id_base) {
            //alledges[nallEdge++] = { std::min(id_potential, stencilIDs[v]), std::max(id_potential, stencilIDs[v]) };
                //alledges[nallEdge++] = { std::min(id_base, stencilIDs[v]), std::max(id_base, stencilIDs[v]) };
            //}
            //if (stencilIDs[v] == id_base && usedStencil == dim) {
            //alledges[nallEdge++] = { std::min(id_potential, stencilIDs[v]), std::max(id_potential, stencilIDs[v]) };
            //}
        //}

        //for (PetscInt v = 0; v < usedStencil+1; ++v) {
      //PetscPrintf(PETSC_COMM_SELF, "edge %" PetscInt_FMT" is %" PetscInt_FMT", %" PetscInt_FMT"\n", v, alledges[v].v1, alledges[v].v2);
    //}

        //PetscReal a[3] = {0.0, 0.0, 0.0}, b[3] = {0.0, 0.0, 0.0};
        //PetscScalar N[3];
        //PetscScalar vol = 0.0;
        //PetscInt edge[2];
        //for (PetscInt i = 0; i < nallEdge; ++i) {
          //edge[0]= alledges[i].v1;
          //edge[1] = alledges[i].v2;
          //PetscScalar vec[dim];
          //for (int d = 0; d < 3; d++) N[d] = 0.0;
          //vec[0] = xCoord[edge[1]] - xCoord[edge[0]];
          //vec[1] = yCoord[edge[1]] - yCoord[edge[0]];
          //N[0] = vec[1];
          //N[1] = -vec[0];

          //PetscScalar midvec[2];
          //midvec[0] = 0.5*(xCoord[edge[1]] + xCoord[edge[0]]);
          //midvec[1] = 0.5*(yCoord[edge[1]] + yCoord[edge[0]]);

          //PetscScalar cellcenter[2] = {xCoord[id_potential], yCoord[id_potential] };
          ////for (PetscInt v = 0; v < usedStencil; ++v) {
            ////cellcenter[0] += xCoord[stencilIDs[v]];
            ////cellcenter[1] += yCoord[stencilIDs[v]];
          ////}
          ////cellcenter[0] /= usedStencil+1;
          ////cellcenter[1] /= usedStencil+1;
          //for (PetscInt v = stencilStart; v < nStencil; ++v) {
      //cellcenter[0] += xCoord[stencilIDs[v]];
      //cellcenter[1] += yCoord[stencilIDs[v]];
      //}
      //cellcenter[0] /= usedStencil+1;
      //cellcenter[1] /= usedStencil+1;

          //PetscScalar centertomid[dim];
          //for (PetscInt d = 0; d < dim; ++d) {
            //centertomid[d] = cellcenter[d] -  midvec[d];
          //}

          //if (N[0]*centertomid[0] + N[1]*centertomid[1] > 0.0) {
            //N[0] = -N[0];
            //N[1] = -N[1];
          //}

          //for (PetscInt d = 0; d < dim; ++d) {
              //for (PetscInt k = 0; k < 2; ++k) {
                  //PetscInt v = edge[k];
                  //if (updatedVertex[v] > 0.5 && updatedVertex[v] < 1.5) {
             //PetscPrintf(PETSC_COMM_SELF, "idwithls is %" PetscInt_FMT" and ls is %.14f\n", v, lsArray[v]);
                      //b[d] += 0.5 * lsArray[v] * N[d];
                  //} else {
                      //a[d] += 0.5 * N[d];
                  //}
              //}
          //}

    //PetscPrintf(PETSC_COMM_SELF,"%" PetscInt_FMT" and %" PetscInt_FMT"\n", edge[0], edge[1]);
    //PetscPrintf(PETSC_COMM_SELF,"%f and %f\n", xCoord[edge[0]], yCoord[edge[0]]);
    //PetscPrintf(PETSC_COMM_SELF,"%f and %f\n", N[0], N[1]);

        //}

        //PetscInt start = alledges[0].v1;  // first vertex of e0
        //PetscInt prev = -1;
        //PetscInt cur = start;

        //PetscInt orderedVerts[usedStencil+1]; // number of vertices = number of unique vertices
        //PetscInt nVerts = 0;
        //orderedVerts[nVerts++] = cur;  // first vertex

        //while (nVerts < usedStencil+1) { // number of unique vertices
            //for (PetscInt i = 0; i < usedStencil+1; ++i) {  // loop over all edges
                //PetscInt v0 = alledges[i].v1;
                //PetscInt v1 = alledges[i].v2;
                //PetscInt next = -1;

                //if (v0 == cur && v1 != prev) next = v1;
                //else if (v1 == cur && v0 != prev) next = v0;

                //if (next != -1) {
                    //orderedVerts[nVerts++] = next;
                    //prev = cur;
                    //cur = next;
                    //break; // move to next vertex
                //}
            //}
        //}

        //PetscScalar temp_area = 0.0;
        //for (PetscInt i = 0; i < nVerts; ++i) {
            //PetscInt j = (i + 1) % nVerts; // next vertex, wrap around
            //PetscInt vi = orderedVerts[i];
            //PetscInt vj = orderedVerts[j];
            //temp_area += xCoord[vi] * yCoord[vj] - xCoord[vj] * yCoord[vi];
        //}

        //vol = 0.5 * PetscAbsReal(temp_area);
    ////PetscPrintf(PETSC_COMM_SELF, "voltest is %f\n", vol);

        //for (PetscInt d = 0; d < dim; ++d) {
       ////PetscPrintf(PETSC_COMM_SELF, "a is %f and b is %f\n", a[d], b[d]);
          //a[d] /= vol;
          //b[d] /= vol;
        //}

        //PetscReal temp_phi = *updatedLS;
         //PetscPrintf(PETSC_COMM_SELF,"Try with %" PetscInt_FMT" stencils -> φ = %f\n", usedStencil, temp_phi);
        //PetscInt temp_result = SolveQuadFormula(dim, a, b, &temp_phi);
         //PetscPrintf(PETSC_COMM_SELF, "Try with %" PetscInt_FMT" stencils -> φ = %f, result=%" PetscInt_FMT"\n", usedStencil, temp_phi, result);
     //PetscPrintf(PETSC_COMM_SELF, "id_potential is %" PetscInt_FMT" and tempphi is %f and result is %" PetscInt_FMT"\n", id_potential, temp_phi, temp_result);
        //if (temp_result != 1) continue;

        //PetscBool monotone = PETSC_TRUE;
        ////PetscReal tol;
    //for (PetscInt v = stencilStart; v < nStencil; ++v) { //for (PetscInt v = 0; v < usedStencil; ++v)
      ////PetscReal phi_n = lsArray[stencilIDs[v]];
      ////tol = 1e-4 * PetscMax(1.0, PetscAbsReal(phi_n));
      //if (PetscAbsReal(temp_phi) < PetscAbsReal(lsArray[stencilIDs[v]])) { //-tol
        //PetscPrintf(PETSC_COMM_SELF, "Rejected φ=%f because neighbor %" PetscInt_FMT" has |φ| smaller.\n", temp_phi, stencilIDs[v]);
                //monotone = PETSC_FALSE;
                //break;
      //}
    //}

    //if (monotone) {
      //accept = PETSC_TRUE;
      //best_phi_local = temp_phi;
    //}

        //if (accept && PetscAbsReal(best_phi_local) < PetscAbsReal(best_phi)) {
          //best_phi = best_phi_local;
          //result = temp_result;
        //}
         //PetscPrintf(PETSC_COMM_SELF, "bestphi is %f\n", best_phi);
      //} //stencil loop

    } // valid base loop
  }

  *updatedLS = best_phi;
  if (rank==1) PetscPrintf(PETSC_COMM_SELF, "id_potential is %" PetscInt_FMT" and ls is %.14f\n", id_potential, *updatedLS);
  //xexit("exit for autofunction");
  return result;
}

// VertexBased_RBF
PetscInt Reconstruction::FFM_VertexBased_RBF(const PetscInt dim, PetscInt id_potential, PetscInt id_base, PetscScalar *updatedVertex, PetscScalar *xCoord, PetscScalar *yCoord, const PetscInt *vertMask, const PetscInt currentLevel, PetscScalar *lsArray, PetscReal *updatedLS) {

  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    PetscInt PnCells, *Pcells;
    DMPlexVertexGetCells(vertDM, vertList[id_potential], &PnCells, &Pcells);

    PetscInt BnCells, *Bcells;
    DMPlexVertexGetCells(vertDM, vertList[id_base], &BnCells, &Bcells);

  PetscInt capacity = 20;
  PetscInt nValid = 2;
  int* vertIDs = (int*)malloc(capacity * sizeof(int));
  vertIDs[0] = id_potential;
  vertIDs[1] = id_base;

  for (PetscInt i = 0; i < PnCells; ++i) {
      for (PetscInt j = 0; j < BnCells; ++j) {
          if (Pcells[i] == Bcells[j]) {
              PetscInt nVert, *verts;
              DMPlexCellGetVertices(vertDM, Pcells[i], &nVert, &verts) >> ablate::utilities::PetscUtilities::checkError;
              for (PetscInt v = 0; v < nVert; ++v) {
                  PetscInt id = reverseVertList[verts[v]];
                  if (updatedVertex[id] > 0.5 && updatedVertex[id] < 1.5 && vertMask[id]==currentLevel-1 && id != id_potential && id != id_base ) {
                      if (nValid >= capacity) {
                          capacity *= 2; // double size
                          vertIDs = (int*)realloc(vertIDs, capacity * sizeof(int));
                      }
                      vertIDs[nValid++] = id;
                  }
              }
          }
      }
  }

  //// Check neighbor vertices of potential vertex to find the ones that has known values including vertices of shared cells
  //PetscInt nVert, *neighborVerts;
  //DMPlexGetNeighbors(vertDM, vertList[id_potential], 1, -1.0, -1, PETSC_TRUE, PETSC_TRUE, &nVert, &neighborVerts); // Return neighboring vertices of a vertex by one level and including the corner ones
  //PetscInt vertIDs[nVert+1], nValid = 1; // nValid is initialized as 1 to include the potential vertex to list
  //vertIDs[0] = id_potential;
  //for (PetscInt nv = 0; nv < nVert; ++nv) {
    //PetscInt index = reverseVertList[neighborVerts[nv]];
    //if (updatedVertex[index] > 0.5 && updatedVertex[index] < 1.5 && vertMask[index]==currentLevel-1) { //should it be in previous level????
      //vertIDs[nValid] = index;
      //++nValid;
    //}
  //}

  //PetscFree(neighborVerts);
  if (nValid-1 < dim+1) return 0;

  PetscInt n = nValid + 3; // 3 is the number of polynomial terms of order 1

  PetscReal RHS[dim][n] = {0};
  Mat A;
  MatCreate(PETSC_COMM_SELF, &A); //for LAPACK, the matrix must be stored entirely on one process, so we have PETSC_COMM_SELF.
  MatSetType(A, MATDENSE); // LAPACK requires MATDENSE
  MatSetSizes(A, n,n,n,n); // the local and global number of rows should be equal since we stored the matrix entirely on one process.
  MatSetUp(A);

  // filling matrix A
  PetscReal r, value, m = 3.0; // m is the power for phs kernel
  PetscReal dx, dy;
  PetscInt vi, vj;
  for (PetscInt i = 0; i < nValid; i++) {
    vi = vertIDs[i];
    for (PetscInt j = 0; j < nValid; j++) {
      vj = vertIDs[j];
      dx = xCoord[vi] - xCoord[vj];
      dy = yCoord[vi] - yCoord[vj];
      r  = PetscSqrtReal(dx*dx + dy*dy);
      value = PetscPowReal(r, m);
      MatSetValue(A, i, j, value, INSERT_VALUES);
    }
  }

  for (PetscInt i = 0; i < nValid; ++i) {
    vi = vertIDs[i];
    MatSetValue(A, i, nValid + 0, 1.0, INSERT_VALUES);
    MatSetValue(A, i, nValid + 1, xCoord[vi], INSERT_VALUES);
    MatSetValue(A, i, nValid + 2, yCoord[vi], INSERT_VALUES);
  }

  for (PetscInt j = 0; j < nValid; ++j) {
    vj = vertIDs[j];
    MatSetValue(A, nValid + 0, j, 1.0, INSERT_VALUES);
    MatSetValue(A, nValid + 1, j, xCoord[vj], INSERT_VALUES);
    MatSetValue(A, nValid + 2, j, yCoord[vj], INSERT_VALUES);
  }

  for (PetscInt ii = 0; ii < 3; ++ii) {
    for (PetscInt jj = 0; jj < 3; ++jj) {
      MatSetValue(A, nValid + ii, nValid + jj, 0.0, INSERT_VALUES);
    }
  }

  MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
  // end of filling matrix A

  // filling right hand sides
  for (PetscInt d = 0; d < dim; ++d) {
    for (PetscInt i = 0; i < nValid; ++i) {
      vi = vertIDs[i];
      dx = xCoord[id_potential] - xCoord[vi];
      dy = yCoord[id_potential] - yCoord[vi];
      r = PetscSqrtReal(dx*dx + dy*dy);
      if (d == 0) {
        RHS[d][i] = m * PetscPowReal(r, m-2) * (xCoord[id_potential]-xCoord[vi]);
      } else {
        RHS[d][i] = m * PetscPowReal(r, m-2) * (yCoord[id_potential]-yCoord[vi]);
      }
    }
  }

  for (PetscInt d = 0; d < dim; ++d) {
    RHS[d][nValid+d+1] = 1.0;
  }
  // end of filling right hand sides

  PetscScalar* Aptr;
  MatDenseGetArray(A,&Aptr);

  PetscBLASInt N = n, nrhs = 1, lda = n, ldb = n, info;
  PetscBLASInt ipiv[n];

  LAPACKgetrf_(&N, &N, Aptr, &lda, ipiv, &info);
  if (info != 0) throw std::runtime_error("GETRF failed");

  for (PetscInt d = 0; d < dim; ++d) {
    LAPACKgetrs_("N", &N, &nrhs, Aptr, &lda, ipiv, RHS[d], &ldb, &info);
    if (info != 0) throw std::runtime_error("GETRS failed");
  }

  // This will create the gradient vector a*phi + b where phi is the level set value to find
  PetscReal a[3] = {0.0, 0.0, 0.0}, b[3] = {0.0, 0.0, 0.0};

  for (PetscInt d = 0; d < dim; ++d) {
    for (PetscInt i = 0; i < nValid; ++i) {
      vi = vertIDs[i];
      if (i == 0) {
        a[d] = a[d] +  RHS[d][i];
      } else {
        b[d] = b[d] +  RHS[d][i] * lsArray[vi];
      }
    }
  }

  PetscInt result = SolveQuadFormula(dim, a, b, updatedLS);
  //PetscPrintf(PETSC_COMM_SELF, "id is %" PetscInt_FMT" and ls is %f and result is %" PetscInt_FMT"\n", id_potential, *updatedLS, result);
  return result;
  //PetscReal vol;
  //DMPlexComputeCellGeometryFVM(vertDM, cellList[10], &vol, NULL, NULL) >> ablate::utilities::PetscUtilities::checkError;
  //for (PetscInt d = 0; d < dim; ++d) {
    //a[d] /= vol;
    //b[d] /= vol;
  //}
}

// VertexBased_ModifiedGreenGauss
PetscInt Reconstruction::FFM_VertexBased_ModifiedGreenGauss(const PetscInt dim, PetscInt id_potential, PetscInt id_base, PetscScalar *updatedVertex, PetscScalar *xCoord, PetscScalar *yCoord, const PetscInt *vertMask, const PetscInt currentLevel, PetscScalar *lsArray, PetscReal *updatedLS) {

  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  // This will create the gradient vector a*phi + b where phi is the level set value to find
  PetscReal a[3] = {0.0, 0.0, 0.0}, b[3] = {0.0, 0.0, 0.0};

  PetscInt vStart, vEnd;
  DMPlexGetDepthStratum(vertDM, 0, &vStart, &vEnd);   // Range of vertices

  PetscInt nFace_base;
  const PetscInt *faces_base;
  DMPlexGetSupportSize(vertDM, vertList[id_base], &nFace_base);
  DMPlexGetSupport(vertDM, vertList[id_base], &faces_base);
  PetscInt baseEdgeVerts[nFace_base], base_nEdgeVerts=0; // vertices that have shared edge with the base vertex not including the base vertex
  for (PetscInt f = 0; f < nFace_base; f++) {
    PetscInt nClosure, *closure = NULL;
    DMPlexGetTransitiveClosure(vertDM, faces_base[f], PETSC_TRUE, &nClosure, &closure) >> ablate::utilities::PetscUtilities::checkError;

    for (PetscInt cl = 0; cl < nClosure * 2; cl += 2) {
      if (closure[cl] >= vStart && closure[cl] < vEnd) {  // Only use the points corresponding to a vertex
        const PetscInt clID = reverseVertList[closure[cl]];
        if (updatedVertex[clID] > 0.5 && updatedVertex[clID] < 1.5 && vertMask[clID] == currentLevel -1 && clID != id_base) {
          baseEdgeVerts[base_nEdgeVerts] = clID;
          ++base_nEdgeVerts;
        }
      }
    }
  }

  // Check neighbor vertices of potential vertex to find the ones that has known values including vertices of shared cells
  PetscInt nVert, *neighborVerts;
  DMPlexGetNeighbors(vertDM, vertList[id_potential], 1, -1.0, -1, PETSC_TRUE, PETSC_TRUE, &nVert, &neighborVerts); // Return neighboring vertices of a vertex by one level and excluding the corner ones
  PetscInt vertIDs[nVert], nValid=0;
  for (PetscInt nv = 0; nv < nVert; ++nv) {
    PetscInt index = reverseVertList[neighborVerts[nv]];
    if (updatedVertex[index] > 0.5 && updatedVertex[index] < 1.5 && vertMask[index]==currentLevel-1) {
      for (PetscInt i = 0; i < base_nEdgeVerts; i++) {
        if ( index == baseEdgeVerts[i]) {
          vertIDs[nValid] = index;
          ++nValid;
        }
      }
      //if (rank==0 && vertList[id_potential]==4000) {
        //PetscPrintf(PETSC_COMM_SELF, "vertid is %" PetscInt_FMT", %f, %f\n", index, xCoord[index], yCoord[index]);
      //}
      //PetscPrintf(PETSC_COMM_SELF, "vertid is %" PetscInt_FMT"\n", vertIDs[nValid]);
    }
  }

  vertIDs[nValid] = id_base; // adding base vertex
  ++nValid;
  PetscFree(neighborVerts);
  if (nValid < dim+1) return 0;

  PetscInt nFace;
  const PetscInt *faces;
  DMPlexGetSupportSize(vertDM, vertList[id_potential], &nFace);
  DMPlexGetSupport(vertDM, vertList[id_potential], &faces);
  PetscInt EdgeVerts[nFace], nEdgeVerts=0; // vertices that have shared edge with the potential vertex
  for (PetscInt f = 0; f < nFace; f++) {
    PetscInt nClosure, *closure = NULL;
    DMPlexGetTransitiveClosure(vertDM, faces[f], PETSC_TRUE, &nClosure, &closure) >> ablate::utilities::PetscUtilities::checkError;

    for (PetscInt cl = 0; cl < nClosure * 2; cl += 2) {
      if (closure[cl] >= vStart && closure[cl] < vEnd) {  // Only use the points corresponding to a vertex
        const PetscInt clID = reverseVertList[closure[cl]];
        //if (rank==1 && vertList[id_potential]==4000) {
            //PetscPrintf(PETSC_COMM_SELF, "idcl is %" PetscInt_FMT", %f, %f\n", clID, xCoord[clID], yCoord[clID]);
        //}
        if (updatedVertex[clID] > 0.5 && updatedVertex[clID] < 1.5 && vertMask[clID] == currentLevel -1 && clID == id_base) { //&& clID < nLocalVert
          EdgeVerts[nEdgeVerts] = clID;
          //if (rank==1 && vertList[id_potential]==4000) {
            //PetscPrintf(PETSC_COMM_SELF, "id is %" PetscInt_FMT", %f, %f\n", clID, xCoord[clID], yCoord[clID]);
          //}
          //PetscPrintf(PETSC_COMM_SELF, "edgevert is %" PetscInt_FMT"\n", EdgeVerts[nEdgeVerts]);
          ++nEdgeVerts;
        }
      }
    }
  }

  // potential vertex
  for (PetscInt n = 0; n < nValid; n++) {
    PetscInt id = vertIDs[n];
    for (PetscInt i = 0; i < nEdgeVerts; i++) {
      if (id != EdgeVerts[i] && id != id_base) {
        PetscReal N[3] = {0.0, 0.0, 0.0};

        PetscReal vx = xCoord[id_potential] - xCoord[id];
        PetscReal vy = yCoord[id_potential] - yCoord[id];

        N[0] = -vy;
        N[1] =  vx;

        // Flip if pointing inward
        PetscReal refx = xCoord[id_potential] - xCoord[id_base];
        PetscReal refy = yCoord[id_potential] - yCoord[id_base];
        PetscReal dot = N[0]*refx + N[1]*refy;
        if (dot < 0) {
          N[0] = -N[0];
          N[1] = -N[1];
        }

        //PetscPrintf(PETSC_COMM_SELF, "id is %" PetscInt_FMT"\n", id);
        for (PetscInt d = 0; d < dim; ++d) {
          a[d] = a[d] + 0.5 * N[d];
          //PetscPrintf(PETSC_COMM_SELF, "b%" PetscInt_FMT" is %f, ls is %f and n%" PetscInt_FMT" is %f\n", d, b[d], lsArray[id], d, N[d]);
          b[d] = b[d] +  0.5 * lsArray[id] * N[d];
          //PetscPrintf(PETSC_COMM_SELF, "b%" PetscInt_FMT" is %f\n", d, b[d]);
        }

      }
    }
  }

  // base vertex
  for (PetscInt n = 0; n < nValid; n++) {
    PetscInt id = vertIDs[n];
    if (id != id_base) {
      PetscReal N[3] = {0.0, 0.0, 0.0};

      PetscReal vx = xCoord[id] - xCoord[id_base];
      PetscReal vy = yCoord[id] - yCoord[id_base];

      N[0] = -vy;
      N[1] =  vx;

      // Flip if pointing inward
      PetscReal refx = xCoord[id_base] - xCoord[id_potential];
      PetscReal refy = yCoord[id_base] - yCoord[id_potential];
      PetscReal dot = N[0]*refx + N[1]*refy;
      if (dot < 0) {
        N[0] = -N[0];
        N[1] = -N[1];
      }

//PetscPrintf(PETSC_COMM_SELF, "id is %" PetscInt_FMT"\n", id);
      for (PetscInt d = 0; d < dim; ++d) {
        //PetscPrintf(PETSC_COMM_SELF, "b%" PetscInt_FMT" is %f, ls is %f and n%" PetscInt_FMT" is %f\n", d, b[d], lsArray[id], d, N[d]);
        b[d] = b[d] +  0.5 * lsArray[id] * N[d];
        //PetscPrintf(PETSC_COMM_SELF, "b%" PetscInt_FMT" is %f and ls is%f\n", d, b[d], lsArray[id_base]);
        b[d] = b[d] +  0.5 * lsArray[id_base] * N[d];
        //PetscPrintf(PETSC_COMM_SELF, "for base, b%" PetscInt_FMT" is %f\n", d, b[d]);
      }
    }

  }

  PetscReal vol;
  DMPlexComputeCellGeometryFVM(vertDM, cellList[10], &vol, NULL, NULL) >> ablate::utilities::PetscUtilities::checkError;
  for (PetscInt d = 0; d < dim; ++d) {
    a[d] /= vol;
    b[d] /= vol;
  }

  PetscInt result = SolveQuadFormula(dim, a, b, updatedLS);
  //return SolveQuadFormula(dim, a, b, updatedLS);
  //PetscPrintf(PETSC_COMM_SELF, "id is %" PetscInt_FMT" and ls is %f and result is %" PetscInt_FMT"\n", id_potential, *updatedLS, result);
  return result;
}

// modified green gauss for vertexbased
// Hybrid FMM
void Reconstruction::FMM_PrimeHybrid(const PetscInt currentLevel, const PetscInt minVerts, const PetscInt *cellMask, const PetscInt *vertMask, Vec updatedVec[2], Vec lsVec[2], Vec lsVecCopy[2], Vec xVec[2], Vec yVec[2], FILE *animFile) {

  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  PetscInt vStart, vEnd;
  DMPlexGetDepthStratum(vertDM, 0, &vStart, &vEnd);   // Range of vertices

  PetscInt dim;
  DMGetDimension(vertDM, &dim);

  MPI_Comm vertComm = PetscObjectComm((PetscObject)(vertDM));

  while (true) {
    // All work must be done on the local vector as a vertex associated with a cell might not be owned by this rank
    //  even if the cell is owned by this rank.
    PetscScalar *lsArray[2] = {nullptr, nullptr};
    VecGetArray(lsVec[LOCAL], &lsArray[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
    VecGetArray(lsVec[GLOBAL], &lsArray[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;

    PetscScalar *updatedVertex[2] = {nullptr, nullptr};
    VecGetArray(updatedVec[LOCAL], &updatedVertex[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
    VecGetArray(updatedVec[GLOBAL], &updatedVertex[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;

    //PetscScalar *lstrue[2] = {nullptr, nullptr};
    //VecGetArray(lsVecCopy[GLOBAL], &lstrue[GLOBAL]) >> utilities::PetscUtilities::checkError;
    //VecGetArray(lsVecCopy[LOCAL], &lstrue[LOCAL]) >> utilities::PetscUtilities::checkError;

    PetscScalar *xCoord[2] = {nullptr, nullptr};
    VecGetArray(xVec[GLOBAL], &xCoord[GLOBAL]) >> utilities::PetscUtilities::checkError;
    VecGetArray(xVec[LOCAL], &xCoord[LOCAL]) >> utilities::PetscUtilities::checkError;

    PetscScalar *yCoord[2] = {nullptr, nullptr};
    VecGetArray(yVec[GLOBAL], &yCoord[GLOBAL]) >> utilities::PetscUtilities::checkError;
    VecGetArray(yVec[LOCAL], &yCoord[LOCAL]) >> utilities::PetscUtilities::checkError;

    PetscInt numVertUpdated_cellbased = 0;
    PetscInt numVertUpdated_vertexbased;
    for (PetscInt c = 0; c < nTotalCell; ++c) {
      if (cellMask[c]==currentLevel) {
        PetscInt cell = cellList[c];
        PetscInt nVert, *verts;
        DMPlexCellGetVertices(vertDM, cell, &nVert, &verts) >> ablate::utilities::PetscUtilities::checkError;

        PetscInt nSetVertices = 0, vertID = -1;
        for (PetscInt v = 0; v < nVert; ++v) {
          PetscInt id = reverseVertList[verts[v]];
          if (updatedVertex[LOCAL][id] > 0.5 && updatedVertex[LOCAL][id] < 1.5) {
            ++nSetVertices;
          }
          else {
            vertID = verts[v]; // When nUpdated+1 == nVert this will contain the ID of the single vertex to be updated
          }
        }

        const PetscInt id = reverseVertList[vertID];

        if (id < nLocalVert && nSetVertices + 1 == nVert) {
          // This will create the gradient vector a*phi + b where phi is the level set value to find
          PetscReal a[3] = {0.0, 0.0, 0.0}, b[3] = {0.0, 0.0, 0.0};

          PetscInt nFace;
          const PetscInt *faces;

          // Get all faces associated with the cell
          DMPlexGetConeSize(vertDM, cell, &nFace) >> ablate::utilities::PetscUtilities::checkError;
          DMPlexGetCone(vertDM, cell, &faces) >> ablate::utilities::PetscUtilities::checkError;
          for (PetscInt f = 0; f < nFace; ++f) {
            PetscReal N[3] = {0.0, 0.0, 0.0};
            DMPlexFaceCentroidOutwardAreaNormal(vertDM, cell, faces[f], NULL, N);

            // All points associated with this face
            PetscInt nClosure, *closure = NULL;
            DMPlexGetTransitiveClosure(vertDM, faces[f], PETSC_TRUE, &nClosure, &closure) >> ablate::utilities::PetscUtilities::checkError;

            PetscReal cnt = 0.0, ave = 0.0, vertCoeff = 0.0;
            for (PetscInt cl = 0; cl < nClosure * 2; cl += 2) {
              if (closure[cl] >= vStart && closure[cl] < vEnd) {  // Only use the points corresponding to a vertex

                const PetscInt clID = reverseVertList[closure[cl]];
                if (closure[cl]==vertID) {
                  //if (updatedVertex[LOCAL][clID] > 0.5 && updatedVertex[LOCAL][clID] < 1.5) throw std::runtime_error("How can this be possible?\n");
          if (updatedVertex[LOCAL][clID] > 0.5 && updatedVertex[LOCAL][clID] < 1.5) {
            PetscPrintf(PETSC_COMM_SELF, "How can this be possible?\n");
            MPI_Abort(PETSC_COMM_WORLD, PETSC_ERR_PLIB); // abort all ranks safely
          }
                    ++vertCoeff;
                }
                else {
                  //if (updatedVertex[LOCAL][clID] < 0.5 || updatedVertex[LOCAL][clID] > 1.5) throw std::runtime_error("How can this be possible?\n");
                    if (updatedVertex[LOCAL][clID] < 0.5 || updatedVertex[LOCAL][clID] > 1.5) {
            PetscPrintf(PETSC_COMM_SELF, "How can this be possible?\n");
            MPI_Abort(PETSC_COMM_WORLD, PETSC_ERR_PLIB); // abort all ranks safely
          }
          //if(rank==1 && vertList[id]==22976)PetscPrintf(PETSC_COMM_SELF, "neighbid is %" PetscInt_FMT", ls is %.14f and vertlitst is %" PetscInt_FMT"\n", clID, lsArray[LOCAL][clID], vertList[id]);
          ave += lsArray[LOCAL][clID];
                }

                cnt += 1.0;
              }
            }

            DMPlexRestoreTransitiveClosure(vertDM, faces[f], PETSC_TRUE, &nClosure, &closure) >> ablate::utilities::PetscUtilities::checkError;

            // Function value at the face center
            ave /= cnt;
            vertCoeff /= cnt;
            for (PetscInt d = 0; d < dim; ++d) {
              a[d] += vertCoeff * N[d];
              b[d] += ave * N[d];
            }
          }

          PetscReal vol;
          DMPlexComputeCellGeometryFVM(vertDM, cell, &vol, NULL, NULL) >> ablate::utilities::PetscUtilities::checkError;
          for (PetscInt d = 0; d < dim; ++d) {
            a[d] /= vol;
            b[d] /= vol;
          }

          PetscScalar resetsign = PetscSignReal(lsArray[GLOBAL][id]);
          PetscInt result = SolveQuadFormula(dim, a, b, &lsArray[GLOBAL][id]);

      //if(rank==1 && vertList[id]==22976)PetscPrintf(PETSC_COMM_SELF, "vertlitst is %" PetscInt_FMT" and %.14f\n", id, lsArray[GLOBAL][id]);

          if (result == 1.0 && updatedVertex[GLOBAL][id] == 0.0) {
            for (PetscInt v = 0; v < nVert; ++v) {
              PetscInt verts_id = reverseVertList[verts[v]];
              if (verts_id != id && PetscAbsReal(lsArray[LOCAL][verts_id])*0.99 > PetscAbsReal(lsArray[GLOBAL][id])) {
                result = 0.0;
                lsArray[GLOBAL][id] = resetsign*PETSC_MAX_REAL;
              }
            }
            updatedVertex[LOCAL][id] = PetscMin(updatedVertex[GLOBAL][id] + result, 1.0);
          }

          updatedVertex[GLOBAL][id] = PetscMin(updatedVertex[GLOBAL][id] + result, 1.0);
          numVertUpdated_cellbased += (updatedVertex[GLOBAL][id] > 0.5);

          // Animation part
          if (updatedVertex[GLOBAL][id] > 0.5) {
              PetscReal x[dim];
              DMPlexComputeCellGeometryFVM(vertDM, vertList[id], NULL, x, NULL) >> ablate::utilities::PetscUtilities::checkError;
              fprintf(animFile, "%d %" PetscInt_FMT" %" PetscInt_FMT" %f %f %s\n", rank, currentLevel, id, x[0], x[1], "cellbased");
              fflush(animFile);
          }

        }
        DMPlexCellRestoreVertices(vertDM, cell, &nVert, &verts) >> ablate::utilities::PetscUtilities::checkError;
      }
    }

    // These update are necessary since we just updated the global vectors in cellbased part
    DMGlobalToLocal(vertDM, updatedVec[GLOBAL], INSERT_VALUES, updatedVec[LOCAL]) >> utilities::PetscUtilities::checkError;
    DMGlobalToLocal(vertDM, lsVec[GLOBAL], INSERT_VALUES, lsVec[LOCAL]) >> utilities::PetscUtilities::checkError;

    //vertexbased
    MPI_Allreduce(MPI_IN_PLACE, &numVertUpdated_cellbased, 1, MPIU_INT, MPIU_SUM, vertComm);

    numVertUpdated_vertexbased = 0;
    if (numVertUpdated_cellbased==0) {
      // Check if there is any vertex to update; maybe we have done all the updates before
      PetscBool anyVertexToUpdate = PETSC_FALSE;
      for (PetscInt v = 0; v < nLocalVert; ++v) {
        if (vertMask[v] == currentLevel && updatedVertex[LOCAL][v] < 0.5) {
          anyVertexToUpdate = PETSC_TRUE;
          break;
        }
      }

      if (anyVertexToUpdate) {
        // finding an unknonw vertex and has a base vertex in previous level with the minimum level set
        PetscInt id_potential = -1;
        PetscInt id_base = -1;
        PetscReal bestPhi = PETSC_MAX_REAL;
        for (PetscInt v = 0; v < nLocalVert; ++v) {
          if (vertMask[v] == currentLevel && updatedVertex[LOCAL][v] < 0.5) {   // not chosen yet
            // find base neighbor
            PetscInt basevertex = -1;
            DMSetBasicAdjacency(vertDM, PETSC_FALSE, PETSC_FALSE); // Connected vertices
            PetscInt nVert = PETSC_DETERMINE;
            PetscInt *neighborVerts = NULL;
            DMPlexGetAdjacency(vertDM, vertList[v], &nVert, &neighborVerts); // just the points that are connected with edge

            PetscReal phibase = PETSC_MAX_REAL;
      for (PetscInt nv = 0; nv < nVert; ++nv) {
        PetscInt index = reverseVertList[neighborVerts[nv]];
        if (vertMask[index] < currentLevel && updatedVertex[LOCAL][index] > 0.5) { // we can have two known vertex in previous level with equal level set, so I choose the first one //vertMask[index] == currentLevel-1
                  if (PetscAbsReal(lsArray[LOCAL][index]) < phibase) {
                    phibase = PetscAbsReal(lsArray[LOCAL][index]);
                    basevertex = neighborVerts[nv];
                  }
        }
      }
            if (basevertex < 0) continue; // no valid base found, skip this candidate

            PetscInt candidateBase = reverseVertList[basevertex];
            PetscReal phiVal = PetscAbsReal(lsArray[LOCAL][candidateBase]);
            PetscReal candidateBaseSign = PetscSignReal(lsArray[LOCAL][candidateBase]);

      // compute global minimum in previous level
      PetscReal phiMin = PETSC_MAX_REAL;
      for (PetscInt i = 0; i < nTotalVert; ++i) { // why total???
        if (vertMask[i] == currentLevel-1 && updatedVertex[LOCAL][i] > 0.5) {
          if (PetscSignReal(lsArray[LOCAL][i]) == candidateBaseSign) {
            PetscReal phicompare = PetscAbsReal(lsArray[LOCAL][i]);
            phiMin = PetscMin(phiMin, phicompare);
          }
        }
      }

      // strict rule: pick only if base is minimum
      if (PetscAbsReal(phiVal - phiMin) < 1e-12 || phiVal <= phiMin) {
        id_potential = v;
        id_base = candidateBase;
        break;  // found the “best” one
      }

      // fallback: track best candidate seen so far
      if (phiVal < bestPhi) {
        id_potential = v;
        id_base = candidateBase;
        bestPhi = phiVal;
      }

          }
    }

        if (id_base > -1 && id_potential > -1) {
          //------------------original vertexbased solver
              //PetscReal x0[dim];
              //DMPlexComputeCellGeometryFVM(vertDM, vertList[id_potential], NULL, x0, NULL) >> ablate::utilities::PetscUtilities::checkError;
              //PetscInt nVert;
          //PetscInt *neighborVerts;
              //DMPlexGetNeighbors(vertDM, vertList[id_potential], 1, -1.0, -1, PETSC_TRUE, PETSC_TRUE, &nVert, &neighborVerts); // Return neighboring vertices of a vertex by one level and including the corner ones
              //const PetscInt minVerts = dim+1;
              //PetscInt result = FFM_VertexBased_Solve(dim, minVerts, x0, nVert, neighborVerts, updatedVertex[LOCAL], lsArray[LOCAL], &lsArray[GLOBAL][id_potential],  vertMask, currentLevel);
              //------------------end of original vertexbased solver

              //------------------modifiedgreengauss vertexbased solver
          //PetscInt result = FFM_VertexBased_ModifiedGreenGauss(dim, id_potential, id_base, updatedVertex[LOCAL], xCoord[LOCAL], yCoord[LOCAL], vertMask, currentLevel, lsArray[LOCAL], &lsArray[GLOBAL][id_potential]);
          //------------------end of modifiedgreengauss vertexbased solver

          //------------------RBF vertexbased solver
          //PetscInt result = FFM_VertexBased_RBF(dim, id_potential, id_base, updatedVertex[LOCAL], xCoord[LOCAL], yCoord[LOCAL], vertMask, currentLevel, lsArray[LOCAL], &lsArray[GLOBAL][id_potential]);
          //------------------end of RBF vertexbased solver

          //------------------generalmodifiedgreengauss vertexbased solver
          PetscInt result = FFM_VertexBased_GMGG(dim, id_potential, id_base, updatedVertex[LOCAL], xCoord[LOCAL], yCoord[LOCAL], vertMask, currentLevel, lsArray[LOCAL], &lsArray[GLOBAL][id_potential]);
          //------------------end of generalmodifiedgreengauss vertexbased solver

      updatedVertex[GLOBAL][id_potential] = PetscMin(updatedVertex[GLOBAL][id_potential] + result, 1.0);
      numVertUpdated_vertexbased  += (updatedVertex[GLOBAL][id_potential] > 0.5);

      if (updatedVertex[GLOBAL][id_potential] > 0.5) {
            updatedVertex[GLOBAL][id_potential] = 0.75;
          }

          //Animation part
          if (updatedVertex[GLOBAL][id_potential] > 0.5) {
              PetscReal x[dim];
              DMPlexComputeCellGeometryFVM(vertDM, vertList[id_potential], NULL, x, NULL) >> ablate::utilities::PetscUtilities::checkError;
              fprintf(animFile, "%d %" PetscInt_FMT" %" PetscInt_FMT" %f %f %s\n", rank, currentLevel, id_potential, x[0], x[1], "vertexbased_v1");
              fflush(animFile);
          }

        }
      }
    }

    VecRestoreArray(lsVec[LOCAL], &lsArray[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
    VecRestoreArray(lsVec[GLOBAL], &lsArray[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;
    DMGlobalToLocal(vertDM, lsVec[GLOBAL], INSERT_VALUES, lsVec[LOCAL]) >> utilities::PetscUtilities::checkError;

    VecRestoreArray(updatedVec[LOCAL], &updatedVertex[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
    VecRestoreArray(updatedVec[GLOBAL], &updatedVertex[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;
    DMGlobalToLocal(vertDM, updatedVec[GLOBAL], INSERT_VALUES, updatedVec[LOCAL]) >> utilities::PetscUtilities::checkError;

    //VecRestoreArray(lsVecCopy[LOCAL], &lstrue[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
    //VecRestoreArray(lsVecCopy[GLOBAL], &lstrue[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;

    MPI_Allreduce(MPI_IN_PLACE, &numVertUpdated_vertexbased, 1, MPIU_INT, MPIU_SUM, vertComm);

    if (numVertUpdated_cellbased==0 && numVertUpdated_vertexbased==0) break;
  }

}

// This implements a FMM-like algorithm to determine a signed distance function given a set of vertices which already have
//  an initial level-set
//  Let a cell with nv-vertices have level-set values at nv-1 vertices. Call the unknown level-set at the last vertex phi.
//  Then it is possible to construct a cell-centered gradient as g = a*phi + b, where a contains the contribution of the unknown level set
//  value and b is the contribution from the nv-1 other vertices. Making g.g==1 results in a quadratic equation, similar to the standard FMM method.
//  For vertices which share multiple possible neighbor cells choose the smallest of the possible results
//
//***********************************************************************************************

// Things to investigate for a paper:
// 1) How the error grows as the level increases
//***********************************************************************************************
void Reconstruction::FMM(const PetscInt *cellMask, const PetscInt *vertMask, Vec lsVec[2]) {

  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  PetscInt dim;
  DMGetDimension(vertDM, &dim);

  Vec updatedVec[2];
  DMGetLocalVector(vertDM, &updatedVec[LOCAL]) >> utilities::PetscUtilities::checkError;
  DMGetGlobalVector(vertDM, &updatedVec[GLOBAL]) >> utilities::PetscUtilities::checkError;
  VecZeroEntries(updatedVec[LOCAL]);
  VecZeroEntries(updatedVec[GLOBAL]);

  PetscScalar *updatedVertex[2] = {nullptr, nullptr}, *lsArr[2] = {nullptr, nullptr};
  VecGetArray(updatedVec[GLOBAL], &updatedVertex[GLOBAL]) >> utilities::PetscUtilities::checkError;
  VecGetArray(updatedVec[LOCAL], &updatedVertex[LOCAL]) >> utilities::PetscUtilities::checkError;
  VecGetArray(lsVec[GLOBAL], &lsArr[GLOBAL]) >> utilities::PetscUtilities::checkError;
  VecGetArray(lsVec[LOCAL], &lsArr[LOCAL]) >> utilities::PetscUtilities::checkError;

  // Declaring a copy of level set vector to check the error later in FMM
  Vec lsVecCopy[2];
  VecDuplicate(lsVec[GLOBAL], &lsVecCopy[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;
  VecDuplicate(lsVec[LOCAL], &lsVecCopy[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
  VecCopy(lsVec[GLOBAL], lsVecCopy[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;
  VecCopy(lsVec[LOCAL], lsVecCopy[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;

  // Declaring coordinate vectors to save coordinates to avoid any issue for finding the coordinates of ghost vertices in other ranks
  Vec xVec[2], yVec[2];
  DMGetLocalVector(vertDM, &xVec[LOCAL]) >> utilities::PetscUtilities::checkError;
  DMGetGlobalVector(vertDM, &xVec[GLOBAL]) >> utilities::PetscUtilities::checkError;
  DMGetLocalVector(vertDM, &yVec[LOCAL]) >> utilities::PetscUtilities::checkError;
  DMGetGlobalVector(vertDM, &yVec[GLOBAL]) >> utilities::PetscUtilities::checkError;

  PetscScalar *xCoord[2] = {nullptr, nullptr}, *yCoord[2] = {nullptr, nullptr};
  VecGetArray(xVec[GLOBAL], &xCoord[GLOBAL]) >> utilities::PetscUtilities::checkError;
  VecGetArray(xVec[LOCAL], &xCoord[LOCAL]) >> utilities::PetscUtilities::checkError;
  VecGetArray(yVec[GLOBAL], &yCoord[GLOBAL]) >> utilities::PetscUtilities::checkError;
  VecGetArray(yVec[LOCAL], &yCoord[LOCAL]) >> utilities::PetscUtilities::checkError;

  for (PetscInt v = 0; v < nLocalVert; ++v) {
    PetscReal x[dim];
    DMPlexComputeCellGeometryFVM(vertDM, vertList[v], NULL, x, NULL) >> utilities::PetscUtilities::checkError; // Get the coordinates of a vertex
    xCoord[LOCAL][v] = x[0];
    xCoord[GLOBAL][v] = x[0];
    yCoord[LOCAL][v] = x[1];
    yCoord[GLOBAL][v] = x[1];
  }

  DMGlobalToLocal(vertDM, xVec[GLOBAL], INSERT_VALUES, xVec[LOCAL]) >> utilities::PetscUtilities::checkError;
  DMGlobalToLocal(vertDM, yVec[GLOBAL], INSERT_VALUES, yVec[LOCAL]) >> utilities::PetscUtilities::checkError;

  //char filename[64];
  //if (rank == 1) {
    //snprintf(filename, sizeof(filename), "coord_rank%" PetscInt_FMT".txt", rank);
    //FILE *file6 = fopen(filename, "w");
    //for (PetscInt v = 0; v < nTotalVert; ++v) {
      //PetscFPrintf(PETSC_COMM_SELF, file6, "%" PetscInt_FMT", %f, %f\n", v, xCoord[LOCAL][v], yCoord[LOCAL][v]);
    //}
  //}
  VecRestoreArray(xVec[GLOBAL], &xCoord[GLOBAL]) >> utilities::PetscUtilities::checkError;
  VecRestoreArray(xVec[LOCAL], &xCoord[LOCAL]) >> utilities::PetscUtilities::checkError;
  VecRestoreArray(yVec[GLOBAL], &yCoord[GLOBAL]) >> utilities::PetscUtilities::checkError;
  VecRestoreArray(yVec[LOCAL], &yCoord[LOCAL]) >> utilities::PetscUtilities::checkError;

  for (PetscInt v = 0; v < nLocalVert; ++v) {
    if (vertMask[v]==1) {
      updatedVertex[LOCAL][v] = 1;
      updatedVertex[GLOBAL][v] = 1;
    }
    else if (vertMask[v]>1) {
      lsArr[LOCAL][v] = PetscSignReal(lsArr[LOCAL][v])*PETSC_MAX_REAL;
      lsArr[GLOBAL][v] = PetscSignReal(lsArr[GLOBAL][v])*PETSC_MAX_REAL;
    }
  }

  for (PetscInt v = nLocalVert; v < nTotalVert; ++v) {
    if (vertMask[v]==1) {
      updatedVertex[LOCAL][v] = 1;
    }
    else if (vertMask[v]>1) {
      lsArr[LOCAL][v] = PetscSignReal(lsArr[LOCAL][v])*PETSC_MAX_REAL;
    }
  }

  VecRestoreArray(updatedVec[GLOBAL], &updatedVertex[GLOBAL]) >> utilities::PetscUtilities::checkError;
  VecRestoreArray(updatedVec[LOCAL], &updatedVertex[LOCAL]) >> utilities::PetscUtilities::checkError;
  VecRestoreArray(lsVec[GLOBAL], &lsArr[GLOBAL]) >> utilities::PetscUtilities::checkError;
  VecRestoreArray(lsVec[LOCAL], &lsArr[LOCAL]) >> utilities::PetscUtilities::checkError;

  char filename[64];
  snprintf(filename, sizeof(filename), "anim_rank%d.txt", rank);
  FILE *animFile = fopen(filename, "w");
  if (!animFile) {
    throw std::runtime_error("Could not open update log file");
  }

  for (PetscInt currentLevel = 2; currentLevel <= nLevels; ++currentLevel) {
  //FILE* animFile = nullptr; // just to avoid creation of any animation file
    Reconstruction::FMM_PrimeHybrid(currentLevel, dim+1, cellMask, vertMask, updatedVec, lsVec, lsVecCopy, xVec, yVec, animFile);

  VecGetArray(updatedVec[GLOBAL], &updatedVertex[GLOBAL]) >> utilities::PetscUtilities::checkError;
  for (PetscInt v = 0; v < nLocalVert; ++v) {
    if (vertMask[v]==currentLevel && updatedVertex[GLOBAL][v]<0.5) {
      PetscReal x[3];
      DMPlexComputeCellGeometryFVM(vertDM, vertList[v], NULL, x, NULL) >> ablate::utilities::PetscUtilities::checkError;
      int rank;
      MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
      PetscPrintf(PETSC_COMM_SELF, "Vertex %" PetscInt_FMT" at (%+f, %+f), level %" PetscInt_FMT", rank %d has not been updated\n", v, x[0], x[1], currentLevel, rank);
      throw std::runtime_error("A vertex has not been updated.\n");
      //MPI_Abort(PETSC_COMM_WORLD, PETSC_ERR_PLIB);
    }
  }
    VecRestoreArray(updatedVec[GLOBAL], &updatedVertex[GLOBAL]) >> utilities::PetscUtilities::checkError;
  }

  fclose(animFile);

  SaveData(vertDM, lsVec[GLOBAL], nLocalVert, vertList, "FMM.txt", 1);

  DMRestoreLocalVector(vertDM, &updatedVec[LOCAL]) >> utilities::PetscUtilities::checkError;
  DMRestoreGlobalVector(vertDM, &updatedVec[GLOBAL]) >> utilities::PetscUtilities::checkError;
}

PetscBool Reconstruction::CutCellfromLS(DM aux_dm, const PetscInt point, const ablate::domain::Field *levelSetField, Vec auxVector) {
  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  PetscInt nv, *verts;
  DMPlexCellGetVertices(aux_dm, point, &nv, &verts);

  PetscScalar *lsArray;
  VecGetArray(auxVector, &lsArray);

  const PetscScalar ZERO_TOL =  1e-15;
  PetscInt nPos = 0, nNeg = 0, nZero = 0;

  for (PetscInt i = 0; i < nv; i++) {
    PetscScalar *lsVal= nullptr;
    DMPlexPointLocalFieldRead(aux_dm, verts[i], levelSetField->id, lsArray, &lsVal);

    if (*lsVal > ZERO_TOL) nPos++;
    else if (*lsVal < -ZERO_TOL) nNeg++;
    else nZero++;
  }

  DMPlexCellRestoreVertices(aux_dm, point, &nv, &verts);
  VecRestoreArray(auxVector, &lsArray);

  return ( (nPos > 0 && nNeg) > 0 || nZero > 0) ? PETSC_TRUE : PETSC_FALSE;
}

//Newton-raphson method
PetscScalar Reconstruction::Newton(const PetscReal* x, void* ctx) {

  EllipseCtx* ellipse = (EllipseCtx*)ctx; // Treat this raw pointer (ctx) as if it points to an EllipseCtx struct.
    PetscScalar a = ellipse->a;
    PetscScalar b = ellipse->b;

  PetscScalar initial_guess[4] = {0.0, PETSC_PI / 2.0, PETSC_PI, 3.0 * PETSC_PI / 2.0};
    PetscScalar best_theta = 0.0;
    PetscScalar min_f = PETSC_MAX_REAL;

  for (PetscInt i = 0; i < 4; ++i) {
        PetscScalar theta = initial_guess[i];
        PetscScalar eps = 1.0;
        PetscInt iter = 0;
        const PetscInt max_iter = 50;
        const PetscScalar tol = 1e-10;
        PetscScalar theta_new, df, d2f;

    while (fabs(eps) > tol && iter < max_iter) {
      Reconstruction::EllipseEq(x, theta, a, b, &df, &d2f);

            if (fabs(d2f) < 1e-14) {
                theta_new = theta - 0.1 * df; // small step instead of df/d2f
            } else {
                theta_new = theta - df / d2f;
            }

      theta_new = theta - df/d2f;
      theta_new = fmod(theta_new, 2.0 * PETSC_PI);
          if (theta_new < 0.0) theta_new += 2.0 * PETSC_PI;

      eps = theta_new - theta;
      theta = theta_new;
      ++iter;
    }

    PetscScalar df_unused;
        PetscScalar f = Reconstruction::EllipseEq(x, theta, a, b, &df_unused, &d2f);

        // Optionally: skip if d2f <= 0 (not a min)
        if (d2f > 0 && f < min_f) {
            min_f = f;
            best_theta = theta;
        }
    }

    if (min_f == PETSC_INFINITY) {
        //throw std::runtime_error("No valid minimum found.\n");
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "No valid minimum found.\n");
    }

    return best_theta;
}

PetscScalar Reconstruction::EllipseEq(const PetscReal* x, PetscScalar theta, PetscScalar a, PetscScalar b, PetscScalar* df, PetscScalar* d2f) {

  *df = 2.0*(-a*sin(theta))*(a*cos(theta)-x[0]) + 2.0*(b*cos(theta))*(b*sin(theta)-x[1]);
  *d2f = 2.0 * ((-a*PetscCosReal(theta)) * (a*PetscCosReal(theta) - x[0]) +
            PetscSqr(a)*PetscSqr(PetscSinReal(theta)) +
            (-b*PetscSinReal(theta)) * (b*PetscSinReal(theta) - x[1]) +
            PetscSqr(b)*PetscSqr(PetscCosReal(theta)));
  PetscScalar f = PetscSqr(a*cos(theta)-x[0]) + PetscSqr(b*sin(theta)-x[1]);

  return f;
}

// This function uses newton-raphson method to find the shortest distance between a point in domain and the elliptic interface
PetscScalar Reconstruction::LSellipse(const PetscReal* x, const PetscInt dim) {
    EllipseCtx ctx;
    ctx.a = 1.0;
    ctx.b = 0.5;

    PetscScalar f_min, df, d2f;

    PetscScalar theta_min = Reconstruction::Newton(x, &ctx);
    f_min = Reconstruction::EllipseEq(x, theta_min, ctx.a, ctx.b, &df, &d2f);
    PetscScalar dist = PetscSqrtReal(f_min);

    PetscScalar sign = (PetscSqr(x[0]/ctx.a) + PetscSqr(x[1]/ctx.b) < 1.0) ? -1.0 : 1.0;
  PetscScalar ls = sign * dist;

  return ls;
}

PetscScalar Reconstruction::NewtonCassiniOval(const PetscReal* x, void* ctx) {
    CassiniOvalCtx* cassinioval = (CassiniOvalCtx*)ctx;
    PetscScalar a = cassinioval->a;
    PetscScalar c = cassinioval->c;

    PetscScalar best_f = PETSC_INFINITY;
    PetscScalar best_theta = 0.0;
    PetscInt best_branch = 0;

    PetscInt N_guess = 100; // Number of initial guesses
    for (PetscInt b = 0; b < 2; b++) {
    for (PetscInt n = 0; n < N_guess; n++) {
      PetscScalar th = n* 2*PETSC_PI/N_guess;
      PetscScalar df, d2f;
      PetscScalar fval = CassiniOvalEq(x, th, a, c, b, &df, &d2f);
      if (fval < best_f) {
        best_f = fval;
                best_theta = th;
                best_branch = b;
            }
        }
  }

  if (best_f == PETSC_INFINITY) {
    //throw std::runtime_error("No valid minimum found.");
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "No valid minimum found.\n");
  }

  PetscInt max_iter = 50;
    PetscScalar tol = 1e-10;

    PetscScalar theta = best_theta;
    PetscInt b = best_branch;
    PetscScalar df_curr, d2f_curr;
    PetscScalar f_curr = CassiniOvalEq(x, theta, a, c, b, &df_curr, &d2f_curr);

    for (PetscInt iter = 0; iter < max_iter; iter++) {
    // check convergence
    if (PetscAbsReal(df_curr) < tol) {
      break;
    }

    // compute step
    PetscScalar step;
    if (d2f_curr > 0.0) {
      step = df_curr / d2f_curr;
    } else {
      step = 0.01 * df_curr;
    }

    PetscScalar theta_new;
    theta_new = theta - step;
    theta_new = fmod(theta_new, 2 * PETSC_PI);

    PetscScalar f_new;
    PetscScalar df_new, d2f_new;
    f_new = CassiniOvalEq(x, theta_new, a, c, b, &df_new, &d2f_new);

    if (f_new < f_curr) {
      theta = theta_new;
      f_curr = f_new;
      df_curr = df_new;
      d2f_curr = d2f_new;
    } else {
      theta_new = theta - 0.5 * step;
      theta_new = fmod(theta_new, 2 * PETSC_PI);
      f_new = CassiniOvalEq(x, theta_new, a, c, b, &df_new, &d2f_new);
      if (f_new < f_curr) {
        theta = theta_new;
        f_curr = f_new;
        df_curr = df_new;
        d2f_curr = d2f_new;
      } else {
        break;
      }
    }
  }

    return f_curr;
}

PetscScalar Reconstruction::CassiniOvalEq(const PetscReal* x, PetscScalar theta, PetscScalar a, PetscScalar c, PetscInt branch, PetscScalar* df, PetscScalar* d2f) {
  PetscScalar s2 = PetscSinReal(2.0 * theta);
  PetscScalar c2 = PetscCosReal(2.0 * theta);
  PetscScalar inner = PetscPowReal(a, 4) - PetscPowReal(c, 4) * s2 * s2;

    if (inner < 0.0) {
        *df = PETSC_INFINITY;
        *d2f = PETSC_INFINITY;
        return PETSC_INFINITY;
    }

    PetscScalar sigma = 1.0;
    PetscScalar u;
    if (branch == 1) {
    u = c * c * c2 + PetscSqrtReal(inner);
  } else {
    u = c * c * c2 - PetscSqrtReal(inner);
    sigma = -1.0;
  }

  if (u < 0) {
    *df = PETSC_INFINITY;
        *d2f = PETSC_INFINITY;
        return PETSC_INFINITY;
  }

  PetscScalar r = PetscSqrtReal(u);
  PetscScalar X = r * PetscCosReal(theta);
  PetscScalar Y = r * PetscSinReal(theta);
  PetscScalar f = (X-x[0])*(X-x[0]) + (Y-x[1])*(Y-x[1]);

  // Derivatives
  PetscScalar delta = PetscSqrtReal(inner);
  PetscScalar S = x[0] * PetscCosReal(theta) + x[1] * PetscSinReal(theta);
  PetscScalar S_prime = -x[0] * PetscSinReal(theta) + x[1] * PetscCosReal(theta);
  PetscScalar u_prime = -2.0*PetscPowReal(c, 2)*s2 - sigma*2*PetscPowReal(c, 4)*s2*c2/delta;
  PetscScalar u_dprime = -4*PetscPowReal(c, 2)*c2 - (4*sigma*PetscPowReal(c, 4)*PetscCosReal(4*theta))/delta - (4*sigma*PetscPowReal(c, 8)*s2*s2*c2*c2)/(PetscPowReal(delta, 3));

    *df = u_prime*(1-S/r) - 2*r*S_prime;
    *d2f = u_dprime*(1-S/r) - 2*u_prime*S_prime/r + S*u_prime*u_prime/(2*r*r*r) + 2*r*S;

    return f;
}

PetscScalar Reconstruction::LScassinioval(const PetscReal* x, const PetscInt dim) {
    CassiniOvalCtx ctx;
    ctx.c = 1.0;
    ctx.a = 1.05;

    PetscScalar f_min = NewtonCassiniOval(x, &ctx);
    PetscScalar dist = PetscSqrtReal(PetscMax(f_min, 0.0)); // ensure positive inside sqrt

    // Sign determination using the implicit Cassini oval equation
    PetscScalar F = ((x[0]-ctx.c)*(x[0]-ctx.c) + x[1]*x[1]) * ((x[0]+ctx.c)*(x[0]+ctx.c) + x[1]*x[1]) - PetscPowReal(ctx.a, 4);
    PetscScalar sign = (F < 0.0) ? -1.0 : 1.0; // Negative inside, positive outside

    return sign * dist;
}

// star---------------------------------------------------------------------------------
PetscScalar Reconstruction::NewtonStar(const PetscReal* x, void* ctx, PetscScalar* theta_star) {
    StarCtx* star = (StarCtx*)ctx;
    PetscScalar a = star->a;
    PetscScalar b = star->b;

    PetscScalar best_f = PETSC_INFINITY;
    PetscScalar best_theta = 0.0;

    // Using 600 points to not get trapped in the wrong lobe's local minimum
    PetscInt N_guess = 600;
    for (PetscInt n = 0; n < N_guess; n++) {
        PetscScalar th = n * 2.0 * PETSC_PI / (PetscScalar)N_guess;
        PetscScalar fval = StarEq(x, th, a, b, nullptr, nullptr);
        if (fval < best_f) {
            best_f = fval;
            best_theta = th;
        }
    }

    PetscScalar theta = best_theta;
    PetscScalar tol = 1e-11;
    PetscInt max_iter = 50;

    for (PetscInt iter = 0; iter < max_iter; iter++) {
        PetscScalar df, d2f;
        StarEq(x, theta, a, b, &df, &d2f);

        if (PetscAbsReal(df) < tol) break;
        // Epsilon prevents division by zero
        PetscScalar step = df / (d2f + 1e-9);
        // Damping factor (0.5) prevents oscillation on sharp diagonals
        PetscScalar theta_new = theta - 0.5 * step;

        if (PetscAbsReal(theta_new - theta) < tol) break;
        theta = theta_new;
    }

    *theta_star = theta;
    return StarEq(x, theta, a, b, nullptr, nullptr);
}

PetscScalar Reconstruction::StarEq(const PetscReal* x, PetscScalar theta, PetscScalar a, PetscScalar b, PetscScalar* df, PetscScalar* d2f) {
    PetscScalar k  = 4.0;
    PetscScalar ct = PetscCosReal(theta);
    PetscScalar st = PetscSinReal(theta);
    PetscScalar ck = PetscCosReal(k*theta);
    PetscScalar sk = PetscSinReal(k*theta);

    // Parametric Radius: r(theta) = b + a*cos(4*theta)
    PetscScalar r   = b + a*ck;
    PetscScalar dr  = -a*k*sk;
    PetscScalar d2r = -a*k*k*ck;

    // Map to Cartesian
    PetscScalar X = r * ct;
    PetscScalar Y = r * st;

    // Squared Distance Function f = (X-x0)^2 + (Y-y0)^2
    PetscScalar f = (X-x[0])*(X-x[0]) + (Y-x[1])*(Y-x[1]);

    // First derivatives of X, Y w.r.t theta
    PetscScalar dX = dr*ct - r*st;
    PetscScalar dY = dr*st + r*ct;

    // Second derivatives of X, Y w.r.t theta
    PetscScalar d2X = d2r*ct - 2.0*dr*st - r*ct;
    PetscScalar d2Y = d2r*st + 2.0*dr*ct - r*st;

    if (df) {
        *df = 2.0*(X - x[0])*dX + 2.0*(Y - x[1])*dY;
    }
    if (d2f) {
        *d2f = 2.0*(dX*dX + (X - x[0])*d2X + dY*dY + (Y - x[1])*d2Y);
    }

    return f;
}

PetscScalar Reconstruction::LSstar(const PetscReal* x, const PetscInt dim) {
    StarCtx ctx;
    ctx.a = 0.3;
    ctx.b = 0.6;

    PetscScalar theta_star;
    PetscScalar f_min = NewtonStar(x, &ctx, &theta_star);
    PetscScalar dist = PetscSqrtReal(PetscMax(f_min, 0.0));

    // Star is defined by: r(theta) = b + a*cos(4theta)
    PetscScalar r_query = PetscSqrtReal(x[0]*x[0] + x[1]*x[1]);
    PetscScalar query_theta = PetscAtan2Real(x[1], x[0]);

    if (r_query < 1e-12) {
        // Center: always inside for star with b > a
        return -(ctx.b - ctx.a);
    }

    // for a star shape, the boundary is single valued in polar coordinates
    // So we can simply check if the point is inside or outside by comparing r with r(theta)
    // However, this only works if the star is star shaped (which it is when b > a)
    PetscScalar r_star_at_angle = ctx.b + ctx.a * PetscCosReal(4.0 * query_theta);
    PetscScalar tol = 1e-12;
    PetscScalar sign;

    if (r_query > r_star_at_angle + tol) {
        sign = 1.0;
    } else if (r_query < r_star_at_angle - tol) {
        sign = -1.0;
    } else {
        // Very close to boundary - use the distance from Newton
        // Points exactly on boundary get sign = 1.0 (outside convention)
        sign = 1.0;
    }

    return sign * dist;
}

PetscScalar Reconstruction::LStwoCircles(const PetscReal* x, const PetscInt dim) {
    const PetscScalar R = 0.5;

    // Circle centers
    const PetscScalar c1[2] = {-1.0, -1.0};
    const PetscScalar c2[2] = { 1.0, 1.0};

    // Distance to circle 1
    PetscScalar d1 = PetscSqrtReal(
        PetscSqr(x[0] - c1[0]) + PetscSqr(x[1] - c1[1])
    ) - R;

    // Distance to circle 2
    PetscScalar d2 = PetscSqrtReal(
        PetscSqr(x[0] - c2[0]) + PetscSqr(x[1] - c2[1])
    ) - R;

    // Union of two circles
    return PetscMin(d1, d2);
}

// This function just uses the known signed distance function of a circle to calculate the shortest distance of each point in domain to the interface
PetscScalar Reconstruction::LScircle(const PetscReal* x, const PetscInt dim) {

  PetscScalar sumSquared = 0.0;
  for (PetscInt i = 0; i < dim; i++) {
    sumSquared += x[i] * x[i];
  }

  PetscScalar ls = PetscSqrtReal(sumSquared) - 1; // Compute the level set at a vertex

  return ls;
}



void Reconstruction::arbit_interface(DM aux_dm, const ablate::domain::Field levelSetField, Vec auxVector) {


  int rank, size;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  MPI_Comm_size(PETSC_COMM_WORLD, &size);


//  DMViewFromOptions(aux_dm, NULL, "-dm_view");
//  Reconstruction_SaveDM(aux_dm, "mesh.txt");


  PetscInt vStart = -1, vEnd= -1;
  DMPlexGetDepthStratum(aux_dm, 0, &vStart, &vEnd) >> ablate::utilities::PetscUtilities::checkError;

  Vec globalVec;
  DMGetGlobalVector(aux_dm, &globalVec);

  PetscInt lsfieldID = levelSetField.id;
  const PetscInt dim = subDomain->GetDimensions();
  PetscReal x[dim]; // coordinates of vertices

  //char filename[64];
  //snprintf(filename, sizeof(filename), "ls_rank%" PetscInt_FMT".txt", rank);
  //FILE *file = fopen(filename, "w");

  PetscScalar *lsArray; // This variable works as a pointer to the level set field in auxiliary vector and the level set field in aux vector is set if we manually set the interface like a simple circle
  VecGetArray(auxVector, &lsArray);

  for (PetscInt v = 0; v < nLocalVert; v++) {
    PetscScalar *lsVal= nullptr;
    DMPlexPointLocalFieldRef(aux_dm, vertList[v], lsfieldID, lsArray, &lsVal); // Access to a specific field variable corresponding to the correct index in aux vector for a specific point in aux dm

    DMPlexComputeCellGeometryFVM(aux_dm, vertList[v], NULL, x, NULL) >> utilities::PetscUtilities::checkError; // Get the coordinates of a vertex
    //*lsVal = LScircle(x, dim);
    //*lsVal = LSellipse(x, dim);
    //*lsVal = LScassinioval(x, dim);
    //*lsVal = LSstar(x, dim);
    *lsVal = LStwoCircles(x, dim);

    //PetscFPrintf(PETSC_COMM_SELF, file, "%" PetscInt_FMT", %f, %f, %.16f\n", v, x[0], x[1], *lsVal);
  }
  //fclose(file);
  VecRestoreArray(auxVector, &lsArray) >> ablate::utilities::PetscUtilities::checkError;

  DMLocalToGlobal(aux_dm, auxVector, INSERT_VALUES, globalVec);
  DMGlobalToLocal(aux_dm, globalVec, INSERT_VALUES, auxVector);
  MPI_Barrier(PETSC_COMM_WORLD);

  PetscScalar *global_lsArray;
  PetscInt Nc = 1;
  PetscMalloc1(nLocalVert * Nc, &global_lsArray);
  PetscScalar *vecArr;
  VecGetArray(auxVector, &vecArr);
  for (PetscInt v = 0; v < nLocalVert; ++v) {
    PetscScalar *val = NULL;
    DMPlexPointLocalFieldRef(aux_dm, vertList[v], lsfieldID, vecArr, &val);
    global_lsArray[v] = *val;
  }
  VecRestoreArray(auxVector, &vecArr);
  SaveData(aux_dm, global_lsArray, nLocalVert, vertList, "lstrue.txt", Nc);
  PetscFree(global_lsArray);


  PetscInt *vertMask = nullptr, *cellMask = nullptr;
  DMGetWorkArray(vertDM, nTotalVert, MPIU_INT, &vertMask) >> ablate::utilities::PetscUtilities::checkError;
  DMGetWorkArray(cellDM, nTotalCell, MPIU_INT, &cellMask) >> ablate::utilities::PetscUtilities::checkError;

  PetscLogDouble t1, t2, elapsed;
  PetscTime(&t1);

  SetMasks(aux_dm, auxVector, levelSetField, nLevels, cellMask, vertMask);

#if 0 // I don't think the lsvec needs to have any values
  // Setting the lsVec which is a vector of level set values for vertices associated with cut-cells
  Vec lsVec[2] = {nullptr, nullptr};                 // [LOCAL, GLOBAL]
  PetscScalar *lsArr[2] =  {nullptr, nullptr};
  DMGetLocalVector(vertDM, &lsVec[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
  DMGetGlobalVector(vertDM, &lsVec[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;
  VecZeroEntries(lsVec[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
  VecZeroEntries(lsVec[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;
  VecGetArray(lsVec[LOCAL], &lsArr[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;

  VecGetArray(auxVector, &lsArray);
  for (PetscInt v = 0; v < nLocalVert; ++v) {
    PetscInt nv, *verts;
    DMPlexCellGetVertices(aux_dm, vertList[v], &nv, &verts);

    for (PetscInt i = 0; i < nv; i++) {
      PetscScalar *lsVal= nullptr;
      DMPlexPointLocalFieldRead(aux_dm, verts[i], lsfieldID, lsArray, &lsVal);

      const PetscInt id = reverseVertList[verts[i]];
      lsArr[LOCAL][id] = *lsVal;
    }
    DMPlexCellRestoreVertices(aux_dm, vertList[v], &nv, &verts);
  }
  VecRestoreArray(auxVector, &lsArray) >> ablate::utilities::PetscUtilities::checkError;
  DMLocalToGlobal(vertDM, lsVec[LOCAL], INSERT_VALUES, lsVec[GLOBAL]);
  DMGlobalToLocal(vertDM, lsVec[GLOBAL], INSERT_VALUES, lsVec[LOCAL]);
  VecRestoreArray(lsVec[LOCAL], &lsArr[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
#endif

  FMM(cellMask, vertMask, lsVec);

  PetscTime(&t2);
  elapsed = t2 - t1;
  PetscPrintf(PETSC_COMM_WORLD,"Elapsed time: %g seconds\n",elapsed);

  PetscPrintf(PETSC_COMM_WORLD, "FMM is done\n");
  MPI_Abort(PETSC_COMM_WORLD, 0);


  //PetscReal *closestPoint;
  //DMGetWorkArray(vertGradDM, nLocalVert*dim, MPIU_REAL, &closestPoint) >> ablate::utilities::PetscUtilities::checkError;
  //PetscInt *cpCell;
  //DMGetWorkArray(vertDM, nLocalVert, MPIU_INT, &cpCell) >> ablate::utilities::PetscUtilities::checkError;

  //InitalizeLevelSet(aux_dm, auxVector, levelSetField, cellMask, vertMask, lsVec, closestPoint, cpCell);
  //SaveData(vertDM, closestPoint, nLocalVert, vertList, "cp.txt", dim);
  //SaveData(vertDM, lsVec[LOCAL], nTotalVert, vertList, "vertLS0_L.txt", 1);
  //SaveData(vertDM, lsVec[GLOBAL], nLocalVert, vertList, "vertLS0_G.txt", 1);

  //FMM(cellMask, vertMask, lsVec);
}


//void Reconstruction::ToLevelSet(DM vofDM, Vec vofVec, const ablate::domain::Field vofField) {

//int rank, size;
//MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
//MPI_Comm_size(PETSC_COMM_WORLD, &size);

//DMViewFromOptions(vofDM, NULL, "-dm_view");
//Reconstruction_SaveDM(vofDM, "mesh.txt");


  //PetscReal         h = 0.0;

  //// Only needed if this is defined over a sub-region of the DM
  //IS subpointIS;
  //const PetscInt* subpointIndices = nullptr;
  //if (subDomain->GetSubAuxDM()!=subDomain->GetAuxDM()) {
    //DMPlexGetSubpointIS(subDomain->GetSubAuxDM(), &subpointIS) >> utilities::PetscUtilities::checkError;
    //ISGetIndices(subpointIS, &subpointIndices) >> utilities::PetscUtilities::checkError;
  //}

  //DMPlexGetMinRadius(vofDM, &h) >> ablate::utilities::PetscUtilities::checkError;
  //h *= 2.0; // Min radius returns the distance between a cell-center and a face. Double it to get the average cell size


  //PetscInt cStart = -1, cEnd = -1;
  //DMPlexGetHeightStratum(cellDM, 0, &cStart, &cEnd) >> ablate::utilities::PetscUtilities::checkError;

  //PetscInt vStart = -1, vEnd = -1;
  //DMPlexGetDepthStratum(vertDM, 0, &vStart, &vEnd) >> ablate::utilities::PetscUtilities::checkError;

///**************** Determine the cut-cells and the initial cell-normal  *************************************/
  //Vec vofGradVec[2] = {nullptr, nullptr};
  //DMGetLocalVector(cellGradDM, &vofGradVec[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
  //DMGetGlobalVector(cellGradDM, &vofGradVec[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;

  //PetscInt *vertMask = nullptr, *cellMask = nullptr;
  //DMGetWorkArray(vertDM, nTotalVert, MPIU_INT, &vertMask) >> ablate::utilities::PetscUtilities::checkError;
  //DMGetWorkArray(cellDM, nTotalCell, MPIU_INT, &cellMask) >> ablate::utilities::PetscUtilities::checkError;


  //SetMasks(vofDM, vofVec, vofField, nLevels, cellMask, vertMask);

//SaveData(cellDM, cellMask, nLocalVert, cellList, "cellMask.txt", 1);
//SaveData(vertDM, vertMask, nLocalVert, vertList, "vertMask.txt", 1);


  //const PetscInt  dim = subDomain->GetDimensions();   // VOF and LS subdomains must have the same dimension. Can't think of a reason they wouldn't.
  //PetscReal *closestPoint;
  //DMGetWorkArray(vertGradDM, nLocalVert*dim, MPIU_REAL, &closestPoint) >> ablate::utilities::PetscUtilities::checkError;
  //PetscInt *cpCell;
  //DMGetWorkArray(vertDM, nLocalVert, MPIU_INT, &cpCell) >> ablate::utilities::PetscUtilities::checkError;

  //InitalizeLevelSet(vofDM, vofVec, vofField, cellMask, vertMask, lsVec, closestPoint, cpCell);
//SaveData(vertDM, closestPoint, nLocalVert, vertList, "cp.txt", dim);

//SaveData(vertDM, lsVec[LOCAL], nTotalVert, vertList, "vertLS0_L.txt", 1);
//SaveData(vertDM, lsVec[GLOBAL], nLocalVert, vertList, "vertLS0_G.txt", 1);

  //FMM(cellMask, vertMask, lsVec);

//  ReinitializeLevelSet(cellMask, vertMask, lsVec);

//SaveData(vertDM, lsVec[LOCAL], nTotalVert, vertList, "vertLS1_L.txt", 1);
//SaveData(vertDM, lsVec[GLOBAL], nLocalVert, vertList, "vertLS1_G.txt", 1);
//xexit("");

  //Vec curv[2];
  //DMGetLocalVector(vertDM, &curv[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
  //DMGetGlobalVector(vertDM, &curv[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;
  //CalculateVertexCurvatures(cellMask, vertMask, lsVec, closestPoint, cpCell, curv);
//SaveData(vertDM, curv[LOCAL], nLocalVert, vertList, "curv0.txt", 1);




//  Smooth(cellMask, vertMask, lsVec, curv);
//SaveData(vertDM, curv[LOCAL], nLocalVert, vertList, "curv1.txt", 1);


  //Extension(cellMask, vertMask, lsVec, closestPoint, cpCell, curv);
//SaveData(vertDM, curv[LOCAL], nLocalVert, vertList, "curv2.txt", 1);



  //DMRestoreLocalVector(cellDM, &curv[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
  //DMRestoreGlobalVector(cellDM, &curv[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;
  //DMRestoreWorkArray(vertDM, nTotalVert, MPIU_INT, &vertMask) >> ablate::utilities::PetscUtilities::checkError;




//  DMRestoreLocalVector(cellGradDM, &cellGradVec[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;
//  DMRestoreGlobalVector(cellGradDM, &cellGradVec[GLOBAL]) >> ablate::utilities::PetscUtilities::checkError;



  //if (subpointIndices) ISRestoreIndices(subpointIS, &subpointIndices) >> utilities::PetscUtilities::checkError;
//xexit("");

// xexit("");

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

//}


