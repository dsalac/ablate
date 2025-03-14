#include "intSharp.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "finiteVolume/processes/twoPhaseEulerAdvection.hpp"
#include "registrar.hpp"
#include "utilities/constants.hpp"
#include "utilities/mathUtilities.hpp"
#include "utilities/petscSupport.hpp"
#include "utilities/petscUtilities.hpp"

#include "finiteVolume/stencils/faceStencilGenerator.hpp"
#include "finiteVolume/stencils/leastSquares.hpp"
#include "finiteVolume/stencils/leastSquaresAverage.hpp"
#include "finiteVolume/stencils/stencil.hpp"


void ablate::finiteVolume::processes::IntSharp::ClearData() {
  if (cellDM) DMDestroy(&cellDM);
  if (cellGradDM) DMDestroy(&cellGradDM);
  if (faceGaussianConv) faceGaussianConv->~GaussianConvolution();
  if (cellGaussianConv) cellGaussianConv->~GaussianConvolution();
}

ablate::finiteVolume::processes::IntSharp::~IntSharp() {
  ablate::finiteVolume::processes::IntSharp::ClearData();
}

void ablate::finiteVolume::processes::IntSharp::Initialize(ablate::finiteVolume::FiniteVolumeSolver &flow) {

  DM dm = flow.GetSubDomain().GetDM();
  PetscInt cStart, cEnd;
  PetscInt dim;

  // Clear any previously allocated memory
  ablate::finiteVolume::processes::IntSharp::ClearData();

  DMGetDimension(dm, &dim) >> utilities::PetscUtilities::checkError;
  DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd) >> utilities::PetscUtilities::checkError;

  ablate::utilities::PetscUtilities::CopyDM(dm, cStart, cEnd, 1, &cellDM);
  ablate::utilities::PetscUtilities::CopyDM(dm, cStart, cEnd, dim, &cellGradDM);

  faceGaussianConv = std::make_shared<ablate::finiteVolume::stencil::GaussianConvolution>(dm, 1.0, 1, ablate::finiteVolume::stencil::GaussianConvolution::DepthOrHeight::HEIGHT);
  cellGaussianConv = std::make_shared<ablate::finiteVolume::stencil::GaussianConvolution>(dm, 1.0, 0, ablate::finiteVolume::stencil::GaussianConvolution::DepthOrHeight::HEIGHT);

}

ablate::finiteVolume::processes::IntSharp::IntSharp(PetscReal epsilon) : epsilon(epsilon) {}


void ablate::finiteVolume::processes::IntSharp::Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) {

  // List of required fields
  std::string fieldList[] = { ablate::finiteVolume::CompressibleFlowFields::GASDENSITY_FIELD,
                              ablate::finiteVolume::CompressibleFlowFields::LIQUIDDENSITY_FIELD,
                              ablate::finiteVolume::CompressibleFlowFields::GASENERGY_FIELD,
                              ablate::finiteVolume::CompressibleFlowFields::LIQUIDENERGY_FIELD,
                              ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD,
                              ablate::finiteVolume::processes::TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD,
                              ablate::finiteVolume::processes::TwoPhaseEulerAdvection::DENSITY_VF_FIELD};

  for (auto field : fieldList) {
    if (!(flow.GetSubDomain().ContainsField(field))) {
      throw std::runtime_error("ablate::finiteVolume::processes::IntSharp expects a "+ field +" field to be defined.");
    }
  }

  // Before each step, compute the source term over the entire dt
//  auto intSharpPreStage = std::bind(&ablate::finiteVolume::processes::IntSharp::IntSharpPreStage, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
//  flow.RegisterPreStage(intSharpPreStage);


  flow.RegisterPreStep([&](TS ts, ablate::solver::Solver &) { ablate::finiteVolume::processes::IntSharp::IntSharpPreStep(ts, flow); });

//  flow.RegisterRHSFunction(ComputeTerm, this);
}

#include <signal.h>



void SaveCellData(DM dm, const Vec vec, const char fname[255], const PetscInt id, PetscInt Nc, ablate::domain::Range range) {


  const PetscScalar *array;
  PetscInt      dim;
  MPI_Comm      comm = PetscObjectComm((PetscObject)dm);
  int rank, size;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  VecGetArrayRead(vec, &array) >> ablate::utilities::PetscUtilities::checkError;
  DMGetDimension(dm, &dim);

  PetscInt boundaryCellStart;
  DMPlexGetCellTypeStratum(dm, DM_POLYTOPE_FV_GHOST, &boundaryCellStart, nullptr) >> ablate::utilities::PetscUtilities::checkError;


  for (PetscInt r = 0; r < size; ++r) {
    if ( rank==r ) {

      FILE *f1;
      if ( rank==0 ) f1 = fopen(fname, "w");
      else f1 = fopen(fname, "a");
      if (f1==nullptr) throw std::runtime_error("Vertex is marked as next to a cut cell but is not!");

      for (PetscInt c = range.start; c < range.end; ++c) {
        PetscInt cell = range.points ? range.points[c] : c;

        DMPolytopeType ct;
        DMPlexGetCellType(dm, cell, &ct) >> ablate::utilities::PetscUtilities::checkError;

        if (ct < 12) {

          PetscReal x0[3];
          DMPlexComputeCellGeometryFVM(dm, cell, nullptr, x0, nullptr) >> ablate::utilities::PetscUtilities::checkError;
          for (PetscInt d = 0; d < dim; ++d) {
            fprintf(f1, "%+.16e\t", x0[d]);
          }

          const PetscScalar *val;
          xDMPlexPointLocalRead(dm, cell, id, array, &val) >> ablate::utilities::PetscUtilities::checkError;
          for (PetscInt i = 0; i < Nc; ++i) {
            fprintf(f1, "%+.16e\t", val[i]);
          }

          fprintf(f1, "\n");
        }
      }
      fclose(f1);
    }

    MPI_Barrier(comm);
  }


  VecRestoreArrayRead(vec, &array) >> ablate::utilities::PetscUtilities::checkError;
}


void ablate::finiteVolume::processes::IntSharp::MemoryHelper(DM dm, VecLoc loc, Vec *vec, PetscScalar **array) {

  if (loc==LOCAL) {
    DMGetLocalVector(dm, vec) >> ablate::utilities::PetscUtilities::checkError;
    VecZeroEntries(*vec) >> ablate::utilities::PetscUtilities::checkError;
    VecGetArray(*vec, array) >> ablate::utilities::PetscUtilities::checkError;
    localVecList.push_back({dm, *vec, *array});
  }
  else {
    DMGetGlobalVector(dm, vec) >> ablate::utilities::PetscUtilities::checkError;
    VecZeroEntries(*vec) >> ablate::utilities::PetscUtilities::checkError;
    VecGetArray(*vec, array) >> ablate::utilities::PetscUtilities::checkError;
    globalVecList.push_back({dm, *vec, *array});
  }

}

void ablate::finiteVolume::processes::IntSharp::MemoryHelper() {
  for (struct vecData data : localVecList) {
    VecRestoreArray(data.vec, &data.array) >> ablate::utilities::PetscUtilities::checkError;
    DMRestoreLocalVector(data.dm, &data.vec) >> ablate::utilities::PetscUtilities::checkError;
  }
  localVecList.clear();

  for (struct vecData data : globalVecList) {
    VecRestoreArray(data.vec, &data.array) >> ablate::utilities::PetscUtilities::checkError;
    DMRestoreGlobalVector(data.dm, &data.vec) >> ablate::utilities::PetscUtilities::checkError;
  }
  globalVecList.clear();


}

void ablate::finiteVolume::processes::IntSharp::SetMask(ablate::domain::Range &cellRange, DM phiDM, Vec phiVec, PetscInt *faceMask) {

  const PetscScalar *phiArray;
  VecGetArrayRead(phiVec, &phiArray) >> ablate::utilities::PetscUtilities::checkError;

  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
    PetscInt cell = cellRange.GetPoint(c);

    const PetscScalar *phiVal;
    DMPlexPointLocalRead(phiDM, cell, phiArray, &phiVal) >> ablate::utilities::PetscUtilities::checkError;

    if (*phiVal > phiRange[0] && *phiVal < phiRange[1]) {

      PetscInt nCells, *cellList;
      DMPlexGetNeighbors(phiDM, cell, 8, -1, -1, PETSC_FALSE, PETSC_FALSE, &nCells, &cellList) >> ablate::utilities::PetscUtilities::checkError;

      for (PetscInt i = 0; i < nCells; ++i) {
        PetscInt nFaces;
        const PetscInt *faces;

        DMPlexGetConeSize(phiDM, cellList[i], &nFaces) >> ablate::utilities::PetscUtilities::checkError;
        DMPlexGetCone(phiDM, cellList[i], &faces) >> ablate::utilities::PetscUtilities::checkError;

        for (PetscInt f = 0; f < nFaces; ++f) {
          faceMask[faces[f]] = 1;
        }
      }

      DMPlexRestoreNeighbors(phiDM, cell, 6, -1, -1, PETSC_FALSE, PETSC_FALSE, &nCells, &cellList) >> ablate::utilities::PetscUtilities::checkError;

    }
  }

  VecRestoreArrayRead(phiVec, &phiArray) >> ablate::utilities::PetscUtilities::checkError;

}

void ablate::finiteVolume::processes::IntSharp::CopyVOFData(TS ts, const ablate::domain::SubDomain& subDomain, ablate::domain::Range cellRange, DM cellDM, Vec phiVec[2], PetscScalar *phiArray[2]) {


  const ablate::domain::Field &phiField = subDomain.GetField(TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD);
  DM solDM = subDomain.GetDM();
  Vec solVec;
  TSGetSolution(ts, &solVec) >> ablate::utilities::PetscUtilities::checkError;

  const PetscScalar *solArray;
  VecGetArrayRead(solVec, &solArray) >> ablate::utilities::PetscUtilities::checkError;

  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
    const PetscInt cell = cellRange.GetPoint(c);

    const PetscReal *solPhi;
    xDMPlexPointLocalRead(solDM, cell, phiField.id, solArray, &solPhi) >> ablate::utilities::PetscUtilities::checkError;

    PetscReal *phi;
    DMPlexPointLocalRef(cellDM, cell, phiArray[GLOBAL], &phi) >> ablate::utilities::PetscUtilities::checkError;
    *phi = *solPhi;
  }
  VecRestoreArrayRead(solVec, &solArray) >> ablate::utilities::PetscUtilities::checkError;
  DMGlobalToLocal(cellDM, phiVec[GLOBAL], INSERT_VALUES, phiVec[LOCAL]) >> ablate::utilities::PetscUtilities::checkError;

}


void ablate::finiteVolume::processes::IntSharp::UpdateSolVec(TS ts, ablate::domain::SubDomain& subDomain, ablate::domain::Range cellRange, DM cellDM, PetscScalar *newPhiArray) {

  const ablate::domain::Field &phiField = subDomain.GetField(ablate::finiteVolume::processes::TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD);
  const ablate::domain::Field &eulerField = subDomain.GetField(ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD);
  const ablate::domain::Field &densityVFField = subDomain.GetField(ablate::finiteVolume::processes::TwoPhaseEulerAdvection::DENSITY_VF_FIELD);

  // Previously computed fields in AUX
  const ablate::domain::Field &gasDensityField = subDomain.GetField(ablate::finiteVolume::CompressibleFlowFields::GASDENSITY_FIELD);
  const ablate::domain::Field &liquidDensityField = subDomain.GetField(ablate::finiteVolume::CompressibleFlowFields::LIQUIDDENSITY_FIELD);
  const ablate::domain::Field &gasEnergyField = subDomain.GetField(ablate::finiteVolume::CompressibleFlowFields::GASENERGY_FIELD);
  const ablate::domain::Field &liquidEnergyField = subDomain.GetField(ablate::finiteVolume::CompressibleFlowFields::LIQUIDENERGY_FIELD);

  const PetscInt dim = subDomain.GetDimensions();

  DM solDM = subDomain.GetDM();
  Vec solVec;
  TSGetSolution(ts, &solVec) >> ablate::utilities::PetscUtilities::checkError;

  PetscScalar *solArray;
  VecGetArray(solVec, &solArray) >> ablate::utilities::PetscUtilities::checkError;

  Vec auxVec = subDomain.GetAuxVector();
  DM auxDM = subDomain.GetAuxDM();
  const PetscScalar *auxArray;
  VecGetArrayRead(auxVec, &auxArray) >> ablate::utilities::PetscUtilities::checkError;

  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
    PetscInt cell = cellRange.GetPoint(c);

    const PetscReal *newPhi;
    DMPlexPointLocalRead(cellDM, cell, newPhiArray, &newPhi) >> ablate::utilities::PetscUtilities::checkError;

    PetscReal *oldPhi;
    xDMPlexPointLocalRef(solDM, cell, phiField.id, solArray, &oldPhi) >> ablate::utilities::PetscUtilities::checkError;


    if (PetscAbsReal(*newPhi - *oldPhi) > PETSC_SMALL) { // Only update those cells where the VOF has changed

      // Pre-computed values from twoPhaseEulerAdvection
      const PetscReal *densityG, *densityL;
      const PetscReal *internalEnergyG, *internalEnergyL;
      xDMPlexPointLocalRead(auxDM, cell, gasDensityField.id, auxArray, &densityG) >> ablate::utilities::PetscUtilities::checkError;
      xDMPlexPointLocalRead(auxDM, cell, liquidDensityField.id, auxArray, &densityL) >> ablate::utilities::PetscUtilities::checkError;
      xDMPlexPointLocalRead(auxDM, cell, gasEnergyField.id, auxArray, &internalEnergyG) >> ablate::utilities::PetscUtilities::checkError;
      xDMPlexPointLocalRead(auxDM, cell, liquidEnergyField.id, auxArray, &internalEnergyL) >> ablate::utilities::PetscUtilities::checkError;

      // Updated VOF
      *oldPhi = *newPhi;

      // Updated density*VOF
      PetscReal *densityVF;
      xDMPlexPointLocalRead(solDM, cell, densityVFField.id, solArray, &densityVF) >> ablate::utilities::PetscUtilities::checkError;
      *densityVF = (*newPhi)*(*densityG);


      PetscReal *euler;
      xDMPlexPointLocalRead(solDM, cell, eulerField.id, solArray, &euler) >> ablate::utilities::PetscUtilities::checkError;

      PetscReal density = euler[ablate::finiteVolume::CompressibleFlowFields::RHO];
      PetscReal ke = 0.0, velocity[3] = {0.0, 0.0, 0.0};
      for (PetscInt d = 0; d < dim; d++) {
        velocity[d] = euler[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density;
        ke += velocity[d]*velocity[d];
      }
      ke *= 0.5;

      // Updated euler field
      euler[ablate::finiteVolume::CompressibleFlowFields::RHO] = (*densityG)*(*newPhi) + (*densityL)*(1 - *newPhi);
      euler[ablate::finiteVolume::CompressibleFlowFields::RHOE] = (*densityG)*(*newPhi)*(*internalEnergyG) + (*densityL)*(1 - *newPhi)*(*internalEnergyL) + ke;
      density = euler[ablate::finiteVolume::CompressibleFlowFields::RHO];
      for (PetscInt d = 0; d < dim; d++) {
        euler[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] = density*velocity[d];
      }
    }
  }
  VecRestoreArray(solVec, &solArray) >> ablate::utilities::PetscUtilities::checkError;
  VecRestoreArrayRead(auxVec, &auxArray) >> ablate::utilities::PetscUtilities::checkError;
}

PetscErrorCode ablate::finiteVolume::processes::IntSharp::IntSharpPreStep(TS flowTS, ablate::solver::Solver &solver) {
  PetscFunctionBegin;
  PetscFunctionReturn(ablate::finiteVolume::processes::IntSharp::IntSharpPreStage(flowTS, solver, 0.0));
}


PetscErrorCode ablate::finiteVolume::processes::IntSharp::IntSharpPreStage(TS flowTS, ablate::solver::Solver &solver, PetscReal stagetime) {
  PetscFunctionBegin;

  const auto &fvSolver = dynamic_cast<ablate::finiteVolume::FiniteVolumeSolver &>(solver);

  StartEvent("IntSharpPreStage");

  const ablate::domain::SubDomain& subDomain = solver.GetSubDomain();

  // Everything in the IntSharp process must be declared, otherwise you get an "invalid use of member ... in static member function error
  PetscInt dim = subDomain.GetDimensions();

  ablate::domain::Range cellRange;
  fvSolver.GetCellRangeWithoutGhost(cellRange);

  PetscReal h;
  PetscCall(DMPlexGetMinRadius(cellDM, &h));
  h *= 2.0; // Min radius returns the distance between a cell-center and a face. Double it to get the average cell sizes

  const PetscReal dt = 0.5*h; // Timestep

  // Copy over the VOF field into a local array so that we can take derivatives
  Vec phiVec[2] = {nullptr, nullptr};
  PetscScalar *phiArray[2] = {nullptr, nullptr};
  MemoryHelper(cellDM, LOCAL, &phiVec[LOCAL], &phiArray[LOCAL]);
  MemoryHelper(cellDM, GLOBAL, &phiVec[GLOBAL], &phiArray[GLOBAL]);
  CopyVOFData(flowTS, subDomain, cellRange, cellDM, phiVec, phiArray);

  PetscInt fStart, fEnd, *faceMask;
  PetscCall(DMPlexGetHeightStratum(cellDM, 1, &fStart, &fEnd));
  PetscCall(DMGetWorkArray(cellDM, fEnd - fStart, MPIU_INT, &faceMask));
  faceMask -= fStart;
  for (PetscInt f = fStart; f < fEnd; ++f) faceMask[f] = 0;

  SetMask(cellRange, cellDM, phiVec[LOCAL], faceMask);

  // check for ghost cells
  DMLabel ghostLabel;
  PetscCall(DMGetLabel(subDomain.GetDM(), "ghost", &ghostLabel));

  const std::shared_ptr<ablate::domain::Region>& solverRegion = fvSolver.GetRegion();
  DMLabel regionLabel;
  PetscInt regionValue;
  ablate::domain::Region::GetLabel(solverRegion, subDomain.GetDM(), regionLabel, regionValue);

  // Get the geometry for the mesh
  Vec cellGeomVec, faceGeomVec;
  fvSolver.GetGeomVecs(&cellGeomVec, &faceGeomVec);

  DM cellGeomDM;
  VecGetDM(cellGeomVec, &cellGeomDM) >> ablate::utilities::PetscUtilities::checkError;
  const PetscScalar* cellGeomArray;
  VecGetArrayRead(cellGeomVec, &cellGeomArray) >> ablate::utilities::PetscUtilities::checkError;
  DM faceGeomDM;
  VecGetDM(faceGeomVec, &faceGeomDM) >> utilities::PetscUtilities::checkError;
  const PetscScalar* faceGeomArray;
  VecGetArrayRead(faceGeomVec, &faceGeomArray) >> utilities::PetscUtilities::checkError;

  PetscInt globalSz;
  VecGetSize(phiVec[GLOBAL], &globalSz) >> ablate::utilities::PetscUtilities::checkError;

char fname[255];
sprintf(fname, "phi_%05d.txt", 0);
SaveCellData(cellDM, phiVec[GLOBAL], fname, -1, 1, cellRange);



//  // Set up the gradient calculator
//std::unique_ptr<stencil::FaceStencilGenerator> faceStencilGenerator;
//if (dim == 1) {
//    faceStencilGenerator = std::make_unique<stencil::LeastSquares>();
//} else {
//    faceStencilGenerator = std::make_unique<stencil::LeastSquaresAverage>();
//}
//std::vector<stencil::Stencil> stencils;
//stencils.resize(fEnd - fStart);
//PetscInt iFace = 0;
//for (PetscInt face = fStart; face < fEnd; face++) {
//    auto& stencil = stencils[iFace++];

//    // make sure that this is a valid face
//    PetscInt ghost, nsupp, nchild;
//    DMLabelGetValue(ghostLabel, face, &ghost) >> utilities::PetscUtilities::checkError;
//    DMPlexGetSupportSize(subDomain.GetDM(), face, &nsupp) >> utilities::PetscUtilities::checkError;
//    DMPlexGetTreeChildren(subDomain.GetDM(), face, &nchild, nullptr) >> utilities::PetscUtilities::checkError;
//    if (ghost >= 0 || nsupp > 2 || nchild > 0) continue;

//    faceStencilGenerator->Generate(face, stencil, subDomain, solverRegion, cellGeomDM, cellGeomArray, faceGeomDM, faceGeomArray);
//}


//Vec smoothPhiVec[2] = {nullptr, nullptr};
//PetscScalar *smoothPhiArray[2] = {nullptr, nullptr};
//MemoryHelper(cellDM, LOCAL, &smoothPhiVec[LOCAL], &smoothPhiArray[LOCAL]);
//MemoryHelper(cellDM, GLOBAL, &smoothPhiVec[GLOBAL], &smoothPhiArray[GLOBAL]);

  for (PetscInt outerIter = 0; outerIter < 1000; ++outerIter) {
//FILE *f1 = fopen("flux.txt", "w");
//FILE *f2 = fopen("smoothPhi.txt","w");
//FILE *f3 = fopen("faceMask.txt", "w");

//    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
//      PetscInt cell = cellRange.GetPoint(c);

//      PetscReal *phi;
//      DMPlexPointLocalRef(cellDM, cell, phiArray[LOCAL], &phi) >> utilities::PetscUtilities::checkError;
//      if (*phi < 0.5) *phi = 0.0;
//      else *phi = 1.0;
//    }

//    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
//      PetscInt cell = cellRange.GetPoint(c);

////      PetscReal *phi;
////      DMPlexPointLocalRef(cellDM, cell, smoothPhiArray[GLOBAL], &phi) >> utilities::PetscUtilities::checkError;
//      PetscReal phi;
//      cellGaussianConv->Evaluate(cell, nullptr, cellDM, -1, phiArray[LOCAL], 0, 1, &phi);
//      PetscReal x[2];
//      DMPlexComputeCellGeometryFVM(cellDM, cell, NULL, x, NULL);

//      fprintf(f2, "%e\t%e\t%e\n", x[0], x[1], phi);

//    }
////    PetscCall(DMGlobalToLocal(cellDM, smoothPhiVec[GLOBAL], INSERT_VALUES, smoothPhiVec[LOCAL]));
//fclose(f2);
//exit(0);



//    PetscInt iStencil = 0;
    for (PetscInt face = fStart; face < fEnd; ++face) {
//      ++iStencil;

//PetscReal x[3];
//DMPlexComputeCellGeometryFVM(cellDM, face, NULL, x, NULL);
//fprintf(f3, "%+e\t%+e\t%ld\n", x[0], x[1], faceMask[face]);



      if (faceMask[face]==1) {

        PetscInt ghost, nsupp, nchild;
        DMLabelGetValue(ghostLabel, face, &ghost) >> utilities::PetscUtilities::checkError;
        DMPlexGetSupportSize(cellDM, face, &nsupp) >> utilities::PetscUtilities::checkError;
        DMPlexGetTreeChildren(cellDM, face, &nchild, nullptr) >> utilities::PetscUtilities::checkError;
        if (ghost >= 0 || nsupp > 2 || nchild > 0) continue;

//        auto& stencil = stencils[iStencil];

        PetscFVFaceGeom* fg;
        DMPlexPointLocalRead(faceGeomDM, face, faceGeomArray, &fg);

        PetscReal smoothPhi = 0.0;
        PetscReal phiGrad[3] = {0.0, 0.0, 0.0}, norm = 0.0;

        for (PetscInt c = 0; c < stencil.stencilSize; c++) {
          PetscInt cell = stencil.stencil[c];
          PetscReal *cellPhi;
          DMPlexPointLocalRead(cellDM, cell, phiArray[LOCAL], &cellPhi) >> utilities::PetscUtilities::checkError;
          smoothPhi += (stencil.weights[c])*(*cellPhi);

          for (PetscInt d = 0; d < dim; ++d) {
            phiGrad[d] += (stencil.gradientWeights[c*dim + d])*(*cellPhi);
          }
        }


//        faceGaussianConv->Evaluate(face, nullptr, cellDM, -1, phiArray[LOCAL], 0, 1, &smoothPhi);

//        for (PetscInt d = 0; d < dim; ++d) {
//          PetscInt dx[3] = {0, 0, 0};
//          dx[d] = 1;
//          faceGaussianConv->Evaluate(face, dx, cellDM, -1, phiArray[LOCAL], 0, 1, &phiGrad[d]);
//        }

//        fprintf(f2,"%+e\t%+e\t%+e\n", fg->centroid[0], fg->centroid[1], smoothPhi);
//        fprintf(f1, "%+e\t%+e\t%+e\t%+e\n", fg->centroid[0], fg->centroid[1], phiGrad[0], phiGrad[1]);

          norm = ablate::utilities::MathUtilities::MagVector(dim, phiGrad) + 1e-12;

          smoothPhi = epsilon*h - smoothPhi*(1-smoothPhi)/norm;
          PetscReal flux[3] = {0.0, 0.0, 0.0};
          for (PetscInt d = 0; d < dim; ++d) {
            flux[d] = phiGrad[d] * smoothPhi;
          }

          fprintf(f1, "%+e\t%+e\t%+e\t%+e\n", fg->centroid[0], fg->centroid[1], flux[0], flux[1]);


          for (PetscInt d = 0; d < dim; ++d) {
            flux[d] *= fg->normal[d];
          }

          // determine where to add the cell values
          const PetscInt* faceCells;
          PetscFVCellGeom *cgL, *cgR;
          DMPlexGetSupport(cellDM, face, &faceCells) >> utilities::PetscUtilities::checkError;
          DMPlexPointLocalRead(cellGeomDM, faceCells[0], cellGeomArray, &cgL) >> utilities::PetscUtilities::checkError;
          DMPlexPointLocalRead(cellGeomDM, faceCells[1], cellGeomArray, &cgR) >> utilities::PetscUtilities::checkError;


           // add the flux back to the cell
          PetscInt cellLabelValue = regionValue;
          DMLabelGetValue(ghostLabel, faceCells[0], &ghost) >> utilities::PetscUtilities::checkError;
          if (regionLabel) {
              DMLabelGetValue(regionLabel, faceCells[0], &cellLabelValue) >> utilities::PetscUtilities::checkError;
          }
          if (ghost <= 0 && regionValue == cellLabelValue) {

            if (faceCells[0] >= globalSz) {
              printf("Out of bounds!\n");
              printf("%s::%s::%d\n", __FILE__, __FUNCTION__, __LINE__);
              exit(0);
            }

            PetscScalar *fL = nullptr;
            DMPlexPointLocalRef(cellDM, faceCells[0], phiArray[GLOBAL], &fL) >> utilities::PetscUtilities::checkError;
            for (PetscInt d = 0; d < dim; ++d) {
              *fL += dt * flux[d] / cgL->volume;
            }
          }

          cellLabelValue = regionValue;
          DMLabelGetValue(ghostLabel, faceCells[1], &ghost) >> utilities::PetscUtilities::checkError;
          if (regionLabel) {
              DMLabelGetValue(regionLabel, faceCells[1], &cellLabelValue) >> utilities::PetscUtilities::checkError;
          }
          if (ghost <= 0 && regionValue == cellLabelValue) {
            if (faceCells[0] >= globalSz) {
              printf("Out of bounds!\n");
              printf("%s::%s::%d\n", __FILE__, __FUNCTION__, __LINE__);
              exit(0);
            }
            PetscScalar *fR = nullptr;
            DMPlexPointLocalRef(cellDM, faceCells[1], phiArray[GLOBAL], &fR) >> utilities::PetscUtilities::checkError;
            for (PetscInt d = 0; d < dim; ++d) {
                *fR -= dt * flux[d] / cgR->volume;
            }
          }

//        }

      }
    }
//exit(0);

    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
      const PetscInt cell = cellRange.GetPoint(c);
      PetscReal *phi;
      DMPlexPointLocalRef(cellDM, cell, phiArray[GLOBAL], &phi) >> utilities::PetscUtilities::checkError;
      *phi = PetscMin(PetscMax(*phi, 0.0), 1.0);
    }

    PetscCall(DMGlobalToLocal(cellDM, phiVec[GLOBAL], INSERT_VALUES, phiVec[LOCAL]));



char fname[255];
sprintf(fname, "phi_%05ld.txt", outerIter+1);
SaveCellData(cellDM, phiVec[GLOBAL], fname, -1, 1, cellRange);
PetscPrintf(PETSC_COMM_WORLD, "%ld\n", outerIter);


  } // End outerIter

//printf("Finished intSharp\n");

  faceMask += fStart;
  DMRestoreWorkArray(cellDM, fEnd - fStart, MPIU_INT, &faceMask);








  // Clear all of the temporary vectors.
  MemoryHelper();

  fvSolver.RestoreRange(cellRange);

PetscPrintf(PETSC_COMM_WORLD, "%s::%s::%d\n", __FILE__, __FUNCTION__, __LINE__);
exit(0);


  EndEvent();

  PetscFunctionReturn(0);
}

REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::IntSharp, "calculates interface regularization term",
         ARG(PetscReal, "epsilon", "epsilon, interface thickness scale parameter (approx. h)")
);
