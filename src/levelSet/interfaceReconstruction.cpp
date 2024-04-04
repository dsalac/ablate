#include "interfaceReconstruction.hpp"
#include <petsc.h>
#include "utilities/petscSupport.hpp"
#include "utilities/mathUtilities.hpp"
#include "domain/fieldAccessor.hpp"


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

  PetscMalloc1(nStencil*nCellList, &interpCellList) >> ablate::utilities::PetscUtilities::checkError;

  for (PetscInt c = 0; c < nCellList; ++c) {

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

          interpCellList[c*nStencil + nQuad*(i*nQuad + j) + k] = interpCell;
        }
      }
    }

    DMPlexRestoreNeighbors(cellDM, cell, 2, -1.0, -1, PETSC_FALSE, PETSC_FALSE, &nCells, &cellList) >> ablate::utilities::PetscUtilities::checkError;

  }

}


//static void Reconstruction_CopyDM(DM oldDM, const PetscInt pStart, const PetscInt pEnd, const PetscInt nDOF, DM *newDM, Vec *localVec, Vec *globalVec) {

//  PetscSection section;


//  // Create a sub auxDM

//  DM coordDM;
//  DMGetCoordinateDM(oldDM, &coordDM) >> ablate::utilities::PetscUtilities::checkError;

//  DMClone(oldDM, newDM) >> ablate::utilities::PetscUtilities::checkError;

//  // this is a hard coded "dmAux" that petsc looks for
//  DMSetCoordinateDM(*newDM, coordDM) >> ablate::utilities::PetscUtilities::checkError;

//  PetscSectionCreate(PetscObjectComm((PetscObject)(*newDM)), &section) >> ablate::utilities::PetscUtilities::checkError;
//  PetscSectionSetChart(section, pStart, pEnd) >> ablate::utilities::PetscUtilities::checkError;
//  for (PetscInt p = pStart; p < pEnd; ++p) PetscSectionSetDof(section, p, nDOF) >> ablate::utilities::PetscUtilities::checkError;
//  PetscSectionSetUp(section) >> ablate::utilities::PetscUtilities::checkError;
//  DMSetLocalSection(*newDM, section) >> ablate::utilities::PetscUtilities::checkError;
//  PetscSectionDestroy(&section) >> ablate::utilities::PetscUtilities::checkError;

//  // Get the vectors
//  DMCreateLocalVector(*newDM, localVec) >> ablate::utilities::PetscUtilities::checkError;
//  DMCreateGlobalVector(*newDM, globalVec) >> ablate::utilities::PetscUtilities::checkError;

//  VecZeroEntries(*localVec);
//  VecZeroEntries(*globalVec);

//}



// The region should be the region WITHOUT ghost cells
Reconstruction::Reconstruction(const std::shared_ptr<ablate::domain::SubDomain> subDomain, std::shared_ptr<domain::Region> region) : region(region), subDomain(subDomain) {

////  int rank;
////  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

//  PetscReal       h = 0.0;
//  const PetscInt  dim = subDomain->GetDimensions();   // VOF and LS subdomains must have the same dimension. Can't think of a reason they wouldn't.
//  const PetscInt  polyAug = 2; // Looks like I need an odd augmented polynomial order for the curvature to be acceptable
//  const bool      doesNotHaveDerivatives = false;
//  const bool      doesNotHaveInterpolation = false;

//  DMPlexGetMinRadius(subDomain->GetDM(), &h) >> ablate::utilities::PetscUtilities::checkError;
//  h *= 2.0; // Min r

//  // Setup the RBF interpolants
//  vertRBF = std::make_shared<ablate::domain::rbf::IMQ>(polyAug, 1e-2*h, doesNotHaveDerivatives, doesNotHaveInterpolation, true);
//  vertRBF->Setup(subDomain);
//  vertRBF->Initialize();


//  cellRBF = std::make_shared<ablate::domain::rbf::MQ>(polyAug, h, doesNotHaveDerivatives, doesNotHaveInterpolation, false);
//  cellRBF->Setup(subDomain);
//  cellRBF->Initialize();


////  PetscSection section;
//  DM subAuxDM = subDomain->GetSubAuxDM();

//  // Create the ranges <--- These might be deleted if they aren't actually needed
//  subDomain->GetRange(nullptr, 0, vertRange);
//  subDomain->GetCellRange(region, cellRange);   // Range of cells without boundary ghosts


//  // Get the point->index mapping for cells
//  reverseVertRange = ablate::domain::ReverseRange(vertRange);
//  reverseCellRange = ablate::domain::ReverseRange(cellRange);


//  // Create individual DMs for vertex- and cell-based data. We need a separate DM for each Vec
//  //    so that we can do global<->local updates on the data that has been updated, rather than
//  //    everything
//  PetscInt vStart, vEnd;
//  DMPlexGetDepthStratum(subAuxDM, 0, &vStart, &vEnd) >> ablate::utilities::PetscUtilities::checkError;
//  Reconstruction_CopyDM(subAuxDM, vStart, vEnd, 1, &vertDM, &vertVec[LOCAL], &vertVec[GLOBAL]);
//  Reconstruction_CopyDM(subAuxDM, vStart, vEnd, dim, &vertGradDM, &vertGradVec[LOCAL], &vertGradVec[GLOBAL]);

//  // Create a DM for vertex-based data
//  PetscInt cStart, cEnd;
//  DMPlexGetHeightStratum(subAuxDM, 0, &cStart, &cEnd) >> ablate::utilities::PetscUtilities::checkError;
//  Reconstruction_CopyDM(subAuxDM, cStart, cEnd, 1, &(Reconstruction::cellDM), &(Reconstruction::cellVec[LOCAL]), &(Reconstruction::cellVec[GLOBAL]));
//  Reconstruction_CopyDM(subAuxDM, cStart, cEnd, dim, &cellGradDM, &cellGradVec[LOCAL], &cellGradVec[GLOBAL]);


//  // Form the list of cells that will have calculations

//  // Get the ghost cell label
//  DMLabel ghostLabel;
//  DMGetLabel(cellDM, "ghost", &ghostLabel) >> utilities::PetscUtilities::checkError;

//  // Get the start of any boundary ghost cells
//  PetscInt boundaryCellStart;
//  DMPlexGetCellTypeStratum(cellDM, DM_POLYTOPE_FV_GHOST, &boundaryCellStart, nullptr) >> utilities::PetscUtilities::checkError;


//  PetscMalloc2(cEnd - cStart, &cellList, cEnd - cStart, &reverseCellList);
//  reverseCellList -= cStart;
//  for (PetscInt c = 0; c < cEnd - cStart; ++c) cellList[c] = -1;

//  nCellList = 0;
//  for (PetscInt c = cStart; c < cEnd; ++c) {

//    reverseCellList[c] = c - cStart;

//    // Check if it's a ghost
//    PetscInt isGhost = -1;
//    if (ghostLabel) {
//        DMLabelGetValue(ghostLabel, c, &isGhost) >> utilities::PetscUtilities::checkError;
//    }

//    // See if it's owned by this rank
//    PetscInt owned;
//    DMPlexGetPointGlobal(cellDM, c, &owned, nullptr) >> utilities::PetscUtilities::checkError;

//    if (owned >= 0 && isGhost < 0 && (boundaryCellStart < 0 || c < boundaryCellStart)) {
//      cellList[nCellList++] = c;
//    }
//  }

////  // Now form the list of vertices.
//  PetscMalloc2(vEnd - vStart, &vertList, vEnd - vStart, &reverseVertList);
//  reverseVertList -= vStart;
//  for (PetscInt v = 0; v < vEnd - vStart; ++v) vertList[v] = -1;

//  nVertList = 0;
//  for (PetscInt v = vStart; v < vEnd; ++v) {

//    reverseVertList[v] = v - vStart;

//    // See if it's owned by this rank
//    PetscInt owned;
//    DMPlexGetPointGlobal(vertDM, v, &owned, nullptr) >> utilities::PetscUtilities::checkError;

//    if (owned >= 0 ) {
//      vertList[nVertList++] = v;
//    }
//  }

//  // Setup the convolution stencil list
//  BuildInterpCellList();


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
    VecDestroy(&(vertVec[i])) >> ablate::utilities::PetscUtilities::checkError;
    VecDestroy(&(vertGradVec[i])) >> ablate::utilities::PetscUtilities::checkError;
    VecDestroy(&(cellVec[i])) >> ablate::utilities::PetscUtilities::checkError;
    VecDestroy(&(cellGradVec[i])) >> ablate::utilities::PetscUtilities::checkError;
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


//      char rankName[255];
//      sprintf(rankName, "%s.%d", fname, rank);
//      f1  = fopen(rankName, "w");

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
          DMPlexPointLocalFieldRead(dm, cell, id, array, &val) >> ablate::utilities::PetscUtilities::checkError;
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



void Reconstruction::ToLevelSet(const ablate::domain::Field vofField) {


Reconstruction_SaveCellData(subDomain->GetFieldDM(vofField), subDomain->GetVec(vofField), "vof.txt", vofField.id, 1, subDomain);
xexit("");

//  const std::shared_ptr<ablate::domain::SubDomain> subDomain = Reconstruction::subDomain;


int rank, size;
MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
MPI_Comm_size(PETSC_COMM_WORLD, &size);



  PetscReal         h = 0.0;
//  const PetscInt    dim = subDomain->GetDimensions();   // VOF and LS subdomains must have the same dimension. Can't think of a reason they wouldn't.
//  PetscInt          *vertMask = nullptr, *cellMask = nullptr;
  DM                vofDM = subDomain->GetFieldDM(vofField);
  Vec               vofVec = subDomain->GetVec(vofField);
  const PetscScalar *vofArray = nullptr;
  PetscScalar       *vertArray = nullptr;
//  PetscScalar       *cellArray = nullptr;


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

//  DM                auxDM = subDomain->GetAuxDM();
//  Vec               auxVec = subDomain->GetAuxVector();
//  const PetscScalar *solArray = nullptr;
//  PetscScalar       *auxArray = nullptr;
//  const PetscInt    lsID = lsField->id, vofID = vofField->id, cellNormalID = cellNormalField->id;


////printf("%+f\n", h);


//  VecGetArrayRead(solVec, &solArray) >> ablate::utilities::PetscUtilities::checkError;
//  VecGetArray(auxVec, &auxArray) >> ablate::utilities::PetscUtilities::checkError;


//  // Pull some work arrays to store the mask information
//  DMGetWorkArray(vertDM, vEnd - vStart, MPIU_INT, &vertMask) >> ablate::utilities::PetscUtilities::checkError;
//  PetscArrayzero(vertMask, vEnd - vStart);
//  vertMask -= vertRange.start; // offset so that we can use start->end

//  DMGetWorkArray(vofDM, cellRange.end - cellRange.start, MPIU_INT, &cellMask) >> ablate::utilities::PetscUtilities::checkError;
//  PetscArrayzero(cellMask, cellRange.end - cellRange.start);
//  cellMask -= cellRange.start; // offset so that we can use start->end


  VecGetArrayRead(vofVec, &vofArray);
  VecGetArray(Reconstruction::vertVec[GLOBAL], &vertArray);


for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {
  const PetscInt vert = vertRange.GetPoint(v);

  PetscInt owned;
  DMPlexGetPointGlobal(subDomain->GetAuxDM(), vert, &owned, nullptr) >> utilities::PetscUtilities::checkError;

  if (owned>=0) {

    PetscReal x[3];
    DMPlexComputeCellGeometryFVM(subDomain->GetAuxDM(), vert, NULL, x, NULL) >> ablate::utilities::PetscUtilities::checkError;
    if (abs(x[0]+0.25)<0.001 && abs(x[1]-0.75)<0.001){
      PetscInt nCells, *cellList;
      DMPlexVertexGetCells(subDomain->GetAuxDM(), vert, &nCells, &cellList);

      PetscScalar smoothVOF = 0.0;

      for (PetscInt i = 0; i < nCells; ++i) {
        const PetscInt globalCell = cellList[i];

        const PetscScalar *vof = nullptr;
        xDMPlexPointLocalRead(vofDM, globalCell, vofField.id, vofArray, &vof);

        smoothVOF += *vof;


      }


      printf("%+f\n",smoothVOF/nCells);

//      for (PetscInt i = 0; i < nCells; ++i) printf("%ld\t", cellList[i]);
//      printf("\n");
      DMPlexVertexRestoreCells(subDomain->GetAuxDM(), vert, &nCells, &cellList);
    }
  }
}



  for (PetscInt v = 0; v < nVertList; ++v) {
    const PetscInt vert = vertList[v];

    PetscScalar smoothVOF = 0.0;

    PetscInt nCells, *cellList;
    DMPlexVertexGetCells(vertDM, vert, &nCells, &cellList);
PetscReal x[3];
DMPlexComputeCellGeometryFVM(vertDM, vert, NULL, x, NULL) >> ablate::utilities::PetscUtilities::checkError;

    for (PetscInt i = 0; i < nCells; ++i) {
      const PetscInt globalCell = subpointIndices ? subpointIndices[cellList[i]] : cellList[i];

      const PetscScalar *vof = nullptr;
      xDMPlexPointLocalRead(vofDM, globalCell, vofField.id, vofArray, &vof);

      smoothVOF += *vof;
//if (abs(x[0]+0.25)<0.001 && abs(x[1]-0.75)<0.001) printf("%+f\t", *vof);
    }

    vertArray[v] = smoothVOF/nCells;


if (abs(x[0]+0.25)<0.001 && abs(x[1]-0.75)<0.001){
    printf("%+f\n", vertArray[v]);
//  for (PetscInt i = 0; i < nCells; ++i) printf("%ld\t", subpointIndices ? subpointIndices[cellList[i]] : cellList[i]);
//  printf("\n");
}


    DMPlexVertexRestoreCells(vertDM, vert, &nCells, &cellList);
  }
  VecRestoreArrayRead(vofVec, &vofArray);

  VecRestoreArray(Reconstruction::vertVec[GLOBAL], &vertArray);

  DMGlobalToLocal(vertDM, vertVec[GLOBAL], INSERT_VALUES, vertVec[LOCAL]) >> utilities::PetscUtilities::checkError;


VecGetArray(Reconstruction::vertVec[GLOBAL], &vertArray);


for (PetscInt r = 0; r < size; ++r) {
  if ( rank==r ) {

    FILE *f1;
    if ( rank==0 ) f1 = fopen("vert0.txt", "w");
    else f1 = fopen("vert0.txt", "a");

    for (PetscInt v = 0; v < nVertList; ++v) {
      const PetscInt vert = vertList[v];
      PetscReal x[3];
      DMPlexComputeCellGeometryFVM(vertDM, vert, NULL, x, NULL) >> ablate::utilities::PetscUtilities::checkError;
      fprintf(f1, "%+f\t%+f\t%+f\n", x[0], x[1], vertArray[v]);
    }
    fclose(f1);
  }

  MPI_Barrier(PETSC_COMM_WORLD);
}

xexit("");

//  Reconstruction::UpdateVec(vertDM, Reconstruction::vertVec_local, Reconstruction::vertVec_global, &vertArray);


//f2 = fopen("vert1.txt","w");
//for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {
//  const PetscInt vert = vertRange.GetPoint(v);
//  PetscReal x[3];
//  DMPlexComputeCellGeometryFVM(vertDM, vert, NULL, x, NULL) >> ablate::utilities::PetscUtilities::checkError;
//  fprintf(f2, "%+f\t%+f\t%+f\n", x[0], x[1], vertArray[v-vertRange.start]);
//}
//fclose(f2);



//  VecGetArray(Reconstruction::cellVec_local, &cellArray);
//Reconstruction::UpdateVec(cellDM, Reconstruction::cellVec_local, Reconstruction::cellVec_global, &cellArray);

//VecSet(Reconstruction::cellVec_global, -1.0);
//DMGlobalToLocal(cellDM, Reconstruction::cellVec_global, INSERT_VALUES, Reconstruction::cellVec_local) >> utilities::PetscUtilities::checkError;

//VecGetArray(Reconstruction::cellVec_local, &cellArray);

//  for (PetscInt c = cellRange.start; c < cellRange.end; ++c){
//    const PetscInt cell = cellRange.GetPoint(c);

//    PetscScalar cellVOF = 0.0;

//    PetscInt nVert, *vertList;
//    DMPlexCellGetVertices(cellDM, cell, &nVert, &vertList);
//    for (PetscInt i = 0; i < nVert; ++i) {
//      PetscInt v = reverseVertRange.GetIndex(vertList[i]);
//      cellVOF += vertArray[v - vertRange.start];
//    }
//    cellArray[c - cellRange.start] = cellVOF/nVert;

//    DMPlexCellRestoreVertices(cellDM, cell, &nVert, &vertList);

//cellArray[c - cellRange.start] = rank;

//  }
//  Reconstruction::UpdateVec(cellDM, Reconstruction::cellVec_local, Reconstruction::cellVec_global, &cellArray);

//{

//ablate::domain::Range cellRange;
//subDomain->GetCellRange(nullptr, cellRange);   // Range of cells without boundary ghosts
//char fname[255];
//sprintf(fname, "loc%d.txt", rank);
//FILE *f1 = fopen(fname,"w");
//for (PetscInt c = cellRange.start; c < cellRange.end; ++c){
//  const PetscInt cell = cellRange.GetPoint(c);
//  PetscReal x[3];
//  DMPlexComputeCellGeometryFVM(cellDM, cell, NULL, x, NULL) >> ablate::utilities::PetscUtilities::checkError;
//  fprintf(f1, "%+f\t%+f\t%+f\n", x[0], x[1], cellArray[c-cellRange.start]);
//}
//fclose(f1);
//}

xexit("");


//VecRestoreArray(workVec, &workArray);
//DMLocalToGlobal(auxDM, workVec, INSERT_VALUES, workVecGlobal) >> utilities::PetscUtilities::checkError;
//DMGlobalToLocal(auxDM, workVecGlobal, INSERT_VALUES, workVec) >> utilities::PetscUtilities::checkError;
//VecGetArray(workVec, &workArray);


//#ifdef saveData
//  sprintf(fname, "vof1_%03ld.txt", saveIter);
//  SaveCellData(auxDM, workVec, fname, vofField, 1, subDomain);
//#endif



///**************** Determine the cut-cells  *************************************/

//  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {

//    PetscInt cell = cellRange.GetPoint(c);

//    if (ablate::levelSet::Utilities::ValidCell(solDM, cell)) {
//      const PetscScalar *vofVal = nullptr;
//      xDMPlexPointLocalRead(solDM, cell, vofID, solArray, &vofVal) >> ablate::utilities::PetscUtilities::checkError;
//      if ( ((*vofVal) > 0.001) && ((*vofVal) < 0.999) ) {
//        cellMask[c] = 1;    // Mark as a cut-cell
//      }
//    }
//  }

//#ifdef saveData
//{
//  sprintf(fname, "mask0_%03ld.txt", saveIter);
//  FILE *f1 = fopen(fname, "w");
//  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
//    const PetscInt cell = cellRange.GetPoint(c);
//    PetscReal x[dim];
//    DMPlexComputeCellGeometryFVM(solDM, cell, NULL, x, NULL) >> ablate::utilities::PetscUtilities::checkError;
//    for (PetscInt d = 0; d < dim; ++d) fprintf(f1, "%+f\t", x[d]);
//    fprintf(f1, "%ld\n", cellMask[c]);
//  }
//  fclose(f1);
//}
//#endif


//// Turn off any "cut cells" where the cell is not surrounded by any other cut cells.
//// To avoid cut-cells two cells-thick turn off any cut-cells which have a neighoring gradient passing through them.
//for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {

//  if (cellMask[c]==1) {

//    const PetscInt cell = cellRange.GetPoint(c);

//    PetscInt nCells, *cells;
//    DMPlexGetNeighbors(solDM, cell, 1, -1.0, -1, PETSC_FALSE, PETSC_FALSE, &nCells, &cells) >> ablate::utilities::PetscUtilities::checkError;

//    PetscInt nCut = 0;
//    for (PetscInt i = 0; i < nCells; ++i) {
//      PetscInt id = reverseCellRange.GetIndex(cells[i]);
//      nCut += (cellMask[id] > 0);
//    }

//    DMPlexRestoreNeighbors(solDM, cell, 1, -1.0, -1, PETSC_FALSE, PETSC_FALSE, &nCells, &cells) >> ablate::utilities::PetscUtilities::checkError;

//    cellMask[c] = (nCut==1 ? 2 : 1); // If nCut equals 1 then the center cell is the only cut cell, so temporarily mark it as 2.


//    const PetscScalar *n = nullptr;
//    xDMPlexPointLocalRead(auxDM, cell, cellNormalID, auxArray, &n) >> ablate::utilities::PetscUtilities::checkError; // VOF normal

//    if (cellMask[c]==1 && ablate::utilities::MathUtilities::MagVector(dim, n)>PETSC_SMALL) {
//      // Now check for two-deep cut-cells.
//      const PetscScalar *n = nullptr;
//      xDMPlexPointLocalRead(auxDM, cell, cellNormalID, auxArray, &n) >> ablate::utilities::PetscUtilities::checkError; // VOF normal


//      const PetscScalar *centerVOF = nullptr;
//      xDMPlexPointLocalRef(auxDM, cell, vofID, workArray, &centerVOF) >> ablate::utilities::PetscUtilities::checkError;

//      const PetscReal direction[2] = {-1.0, +1.0};
//      for (PetscInt d = 0; d < 2; ++d) {
//        PetscInt neighborCell = -1;
//        DMPlexGetForwardCell(auxDM, cell, n, direction[d], &neighborCell) >> ablate::utilities::PetscUtilities::checkError;
//        if (neighborCell > -1) {

//          const PetscReal *neighborVOF;
//          xDMPlexPointLocalRef(auxDM, neighborCell, vofID, workArray, &neighborVOF) >> ablate::utilities::PetscUtilities::checkError;

//          if (PetscAbsReal(*neighborVOF - 0.5) < PetscAbsReal(*centerVOF - 0.5)) {
//            cellMask[c] = 2;
//            break;
//          }
//        }
//      }
//    }
//  }
//}

//for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
//  cellMask[c] = (cellMask[c]==2 ? 0 : cellMask[c]);
//}

///**************** Determine the initial unit normal *************************************/


//  DM cutCellDM = solDM;
//  Vec cutCellVec = solVec;

//  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {

//    PetscInt cell = cellRange.GetPoint(c);

//    PetscScalar *n = nullptr;
//    xDMPlexPointLocalRef(auxDM, cell, cellNormalID, auxArray, &n);
//    for (PetscInt d = 0; d < dim; ++d ) n[d] = 0.0;

//    if (cellMask[c]==1) {

//      // Will this crap near the edges of a processor?
//      DMPlexCellGradFromCell(cutCellDM, cell, cutCellVec, vofID, 0, n);
////      if ( dim > 0 ) n[0] = cellRBF->EvalDer(auxDM, workVec, vofID, cell, 1, 0, 0);
////      if ( dim > 1 ) n[1] = cellRBF->EvalDer(auxDM, workVec, vofID, cell, 0, 1, 0);
////      if ( dim > 2 ) n[2] = cellRBF->EvalDer(auxDM, workVec, vofID, cell, 0, 0, 1);

//      ablate::utilities::MathUtilities::NormVector(dim, n);
//      for (PetscInt d = 0; d < dim; ++d) n[d] *= -1.0;

//      // Mark all vertices of this cell as associated with a cut-cell
//      PetscInt nv, *verts;
//      DMPlexCellGetVertices(solDM, cell, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;
//      for (PetscInt v = 0; v < nv; ++v) {
//        PetscInt vert_i = reverseVertRange.GetIndex(verts[v]);
//        vertMask[vert_i] = 1;
//      }
//      DMPlexCellRestoreVertices(solDM, cell, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;
//    }
//  }


//  subDomain->UpdateAuxLocalVector();



//#ifdef saveData
//{
//  sprintf(fname, "mask1_%03ld.txt", saveIter);
//  FILE *f1 = fopen(fname, "w");
//  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
//    const PetscInt cell = cellRange.GetPoint(c);
//    PetscReal x[dim];
//    DMPlexComputeCellGeometryFVM(solDM, cell, NULL, x, NULL) >> ablate::utilities::PetscUtilities::checkError;
//    for (PetscInt d = 0; d < dim; ++d) fprintf(f1, "%+f\t", x[d]);
//    fprintf(f1, "%ld\n", cellMask[c]);
//  }
//  fclose(f1);
//  sprintf(fname, "cellNormal0_%03ld.txt", saveIter);
//  SaveCellData(auxDM, auxVec, fname, cellNormalField, dim, subDomain);
//}
//#endif







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


