#include "intSharp.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "finiteVolume/processes/twoPhaseEulerAdvection.hpp"
#include "utilities/constants.hpp"
#include "utilities/mathUtilities.hpp"
#include "utilities/petscSupport.hpp"
#include "utilities/petscUtilities.hpp"
#include "twoPhaseEulerAdvection.hpp"
#include "finiteVolume/stencils/faceStencilGenerator.hpp"
#include "finiteVolume/stencils/leastSquares.hpp"
#include "finiteVolume/stencils/leastSquaresAverage.hpp"
#include "finiteVolume/stencils/stencil.hpp"


void ablate::finiteVolume::processes::IntSharp::ClearData() {
  if (faceGaussianConv) faceGaussianConv->~GaussianConvolution();
}

ablate::finiteVolume::processes::IntSharp::~IntSharp() {
  ablate::finiteVolume::processes::IntSharp::ClearData();
}


// Every time the mesh changes
void ablate::finiteVolume::processes::IntSharp::Initialize(ablate::finiteVolume::FiniteVolumeSolver &flow) {


  ablate::domain::SubDomain& subDomain = flow.GetSubDomain();
  DM dm = subDomain.GetDM();

  faceGaussianConv = std::make_shared<ablate::finiteVolume::stencil::GaussianConvolution>(dm, 2.0, 1, ablate::finiteVolume::stencil::GaussianConvolution::DepthOrHeight::HEIGHT);

  // Set up the gradient calculator
  std::unique_ptr<stencil::FaceStencilGenerator> faceStencilGenerator;
  if (subDomain.GetDimensions() == 1) {
      faceStencilGenerator = std::make_unique<stencil::LeastSquares>();
  } else {
      faceStencilGenerator = std::make_unique<stencil::LeastSquaresAverage>();
  }

  Vec cellGeomVec, faceGeomVec;
  flow.GetGeomVecs(&cellGeomVec, &faceGeomVec);

  DM faceDM, cellDM;
  VecGetDM(faceGeomVec, &faceDM) >> utilities::PetscUtilities::checkError;
  VecGetDM(cellGeomVec, &cellDM) >> utilities::PetscUtilities::checkError;

  const PetscScalar *cellGeomArray, *faceGeomArray;
  VecGetArrayRead(cellGeomVec, &cellGeomArray) >> utilities::PetscUtilities::checkError;
  VecGetArrayRead(faceGeomVec, &faceGeomArray) >> utilities::PetscUtilities::checkError;

  // perform some fv and mpi ghost cell checks
  PetscInt gcStart;
  DMPlexGetCellTypeStratum(dm, DM_POLYTOPE_FV_GHOST, &gcStart, nullptr) >> utilities::PetscUtilities::checkError;

  // check for ghost cells
  DMLabel ghostLabel;
  DMGetLabel(dm, "ghost", &ghostLabel) >> utilities::PetscUtilities::checkError;

  const std::shared_ptr<domain::Region> solverRegion = flow.GetRegion();


  // Size up the stencil
//  PetscInt fStart, fEnd;
//  DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd) >> utilities::PetscUtilities::checkError;
//  stencils.resize(fEnd - fStart);

//  // Compute the stencil for each face
//  PetscInt iFace = 0;
//  for (PetscInt face = fStart; face < fEnd; face++) {
//      auto& stencil = stencils[iFace++];

//      // make sure that this is a valid face
//      PetscInt ghost, nsupp, nchild;
//      DMLabelGetValue(ghostLabel, face, &ghost) >> utilities::PetscUtilities::checkError;
//      DMPlexGetSupportSize(dm, face, &nsupp) >> utilities::PetscUtilities::checkError;
//      DMPlexGetTreeChildren(dm, face, &nchild, nullptr) >> utilities::PetscUtilities::checkError;
//      if (ghost >= 0 || nsupp > 2 || nchild > 0) continue;

//      faceStencilGenerator->Generate(face, stencil, subDomain, solverRegion, cellDM, cellGeomArray, faceDM, faceGeomArray);
//  }
  // clean up the geom
  VecRestoreArrayRead(cellGeomVec, &cellGeomArray) >> utilities::PetscUtilities::checkError;
  VecRestoreArrayRead(faceGeomVec, &faceGeomArray) >> utilities::PetscUtilities::checkError;

}

ablate::finiteVolume::processes::IntSharp::IntSharp(PetscReal Gamma, PetscReal epsilonIn) : Gamma(Gamma), epsilonIn(epsilonIn) {}

// Run once per simulation
void ablate::finiteVolume::processes::IntSharp::Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) {

  // List of required fields
  std::string fieldList[] = { ablate::finiteVolume::CompressibleFlowFields::LIQUIDDENSITY_FIELD,
                              ablate::finiteVolume::processes::TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD,
                              ablate::finiteVolume::processes::TwoPhaseEulerAdvection::DENSITY_VF_FIELD
                              };

  ablate::domain::FieldLocation locationList[] = {
                              ablate::domain::FieldLocation::AUX,
                              ablate::domain::FieldLocation::SOL,
                              ablate::domain::FieldLocation::SOL
                              };

  PetscInt i = 0;
  for (auto fieldName : fieldList) {
    if (!(flow.GetSubDomain().ContainsField(fieldName))) {
      throw std::runtime_error("ablate::finiteVolume::processes::IntSharp expects a "+ fieldName +" field to be defined.");
    }
    const ablate::domain::Field field = flow.GetSubDomain().GetField(fieldName);
    if (field.location != locationList[i++]) {
      throw std::runtime_error("ablate::finiteVolume::processes::IntSharp: "+ fieldName +" is in the incorrect location.");
    }
  }


  PetscReal h;
  DMPlexGetMinRadius(flow.GetSubDomain().GetDM(), &h);
  h *= 2.0; // Min radius returns the distance between a cell-center and a face. Double it to get the average cell sizes
  epsilon = epsilonIn*h;


  flow.RegisterRHSFunction(IntSharpFlux,
                         this,
                         ablate::finiteVolume::processes::TwoPhaseEulerAdvection::DENSITY_VF_FIELD,
                         {ablate::finiteVolume::processes::TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD, ablate::finiteVolume::processes::TwoPhaseEulerAdvection::DENSITY_VF_FIELD},
                         {ablate::finiteVolume::CompressibleFlowFields::LIQUIDDENSITY_FIELD});


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

//FILE *f1 = fopen("flux.txt", "w");

//static PetscInt cnt = 0;

PetscErrorCode ablate::finiteVolume::processes::IntSharp::IntSharpFlux(PetscInt dim, const PetscFVFaceGeom* fg,
      const PetscInt uOff[], const PetscInt uOff_x[],
      const PetscScalar field[], const PetscScalar grad[],
      const PetscInt aOff[], const PetscInt aOff_x[],
      const PetscScalar aux[], const PetscScalar gradAux[],
      PetscScalar flux[], void* ctx) {

    PetscFunctionBeginUser;

    ablate::finiteVolume::processes::IntSharp *process = (ablate::finiteVolume::processes::IntSharp *)ctx;
    const PetscReal aG = field[uOff[0]];

    flux[0] = 0.0;

//    const PetscReal dist = ablate::utilities::MathUtilities::MagVector(dim, fg->centroid);

//    if (dist < 0.045) {

    if (aG >= process->phiRange[0] && aG <= process->phiRange[1]) {

//if (PetscAbsReal(fg->centroid[0]+0.0252475247524753) < 1e-6 && PetscAbsReal(fg->centroid[1])<1e-6) {
//  printf("%ld: %e\n", cnt, aG);
//}
//if (PetscAbsReal(fg->centroid[0]+0.0143564356435644) < 1e-6 && PetscAbsReal(fg->centroid[1])<1e-6) {
//  printf("%ld: %e\n", cnt, aG);
//}

//      const PetscReal densityVF = field[uOff[1]];
      const PetscReal *gAG = grad + uOff_x[0];
//      const PetscReal liquidDensity = aux[aOff[0]];
      const PetscReal epsilon = process->epsilon;
      const PetscReal Gamma = process->Gamma;
      const PetscReal nrm = ablate::utilities::MathUtilities::MagVector(dim, gAG);

      flux[0] = ablate::utilities::MathUtilities::DotVector(dim, gAG, fg->normal);

//      const PetscReal fac = -liquidDensity*Gamma*(epsilon - aG*(1-aG)/nrm);
      const PetscReal fac = -998.23*Gamma*(epsilon - aG*(1-aG)/nrm);

      flux[0] *= fac;

//if (cnt==12776 || cnt==13887) {
//  printf("%e: %+e\n", aG, flux[0]);
//}


//fprintf(f1, "%+e\t%+e\t%+e\t%+e\n", fg->centroid[0], fg->centroid[1], fac*gAG[0], fac*gAG[1]);

//FILE *f1 = fopen("data.txt", "a");

//fprintf(f1, "%+e\t%+e\t%+e\t%+e\t%+e\t%+e\n", fg->centroid[0], fg->centroid[1], alphaG, liquidDensity, gradAlphaG[0], gradAlphaG[1]);

//fclose(f1);

    }
//++cnt;
//printf("%s::%s::%d\n", __FILE__, __FUNCTION__, __LINE__);
//exit(0);

    PetscFunctionReturn(0);
}



//static PetscInt cnt = 0;
// Note: locX is a local vector, which means it contains all overlap cells
PetscErrorCode ablate::finiteVolume::processes::IntSharp::ComputeTerm(const FiniteVolumeSolver &solver, DM dm, PetscReal time, Vec locX, Vec locFVec, void *ctx) {

  PetscFunctionBegin;

  ablate::finiteVolume::processes::IntSharp *process = (ablate::finiteVolume::processes::IntSharp *)ctx;

  const ablate::domain::SubDomain& subDomain = solver.GetSubDomain();

  // Everything in the IntSharp process must be declared, otherwise you get an "invalid use of member ... in static member function error
  PetscInt dim = subDomain.GetDimensions();

  const ablate::domain::Field &phiField = subDomain.GetField(TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD);
  const ablate::domain::Field &liquidDensityField = subDomain.GetField(ablate::finiteVolume::CompressibleFlowFields::LIQUIDDENSITY_FIELD);
  const ablate::domain::Field &densityVFField = subDomain.GetField(TwoPhaseEulerAdvection::DENSITY_VF_FIELD);

  DM auxDM = subDomain.GetAuxDM();
  Vec auxVec = subDomain.GetAuxVector();
  const PetscScalar *auxArray;
  VecGetArrayRead(auxVec, &auxArray) >> utilities::PetscUtilities::checkError;

  DMLabel regionLabel;
  PetscInt regionValue;
  ablate::domain::Region::GetLabel(solver.GetRegion(), dm, regionLabel, regionValue);

  DMLabel ghostLabel;
  DMGetLabel(dm, "ghost", &ghostLabel) >> utilities::PetscUtilities::checkError;

  Vec cellGeomVec, faceGeomVec;
  solver.GetGeomVecs(&cellGeomVec, &faceGeomVec);

  DM faceDM, cellDM;
  VecGetDM(faceGeomVec, &faceDM) >> utilities::PetscUtilities::checkError;
  VecGetDM(cellGeomVec, &cellDM) >> utilities::PetscUtilities::checkError;

  const PetscScalar *cellGeomArray, *faceGeomArray;
  VecGetArrayRead(cellGeomVec, &cellGeomArray) >> utilities::PetscUtilities::checkError;
  VecGetArrayRead(faceGeomVec, &faceGeomArray) >> utilities::PetscUtilities::checkError;


  const PetscScalar *locXArray;
  VecGetArrayRead(locX, &locXArray) >> utilities::PetscUtilities::checkError;

  PetscScalar *locFArray;
  VecGetArray(locFVec, &locFArray) >> utilities::PetscUtilities::checkError;

  ablate::domain::Range faceRange;
  solver.GetFaceRange(faceRange);



//  const PetscReal *phiRange = process->phiRange;
//  const PetscReal epsilon = process->epsilon;
//  const PetscReal Gamma = process->Gamma;

  std::shared_ptr<ablate::finiteVolume::stencil::GaussianConvolution> faceGaussianConv = process->faceGaussianConv;


//  char fname[244];
//  sprintf(fname, "flux_%ld.txt", cnt++);
//  FILE *f1 = fopen(fname, "w");

//  std::vector<stencil::Stencil> stencils = process->stencils;
//  PetscInt iFace = 0;
  for (PetscInt f = faceRange.start; f < faceRange.end; ++f) {
    const PetscInt face = faceRange.GetPoint(f);

//    auto& stencil = stencils[iFace++];

    // make sure that this is a valid face
    PetscInt ghost, nsupp, nchild;
    DMLabelGetValue(ghostLabel, face, &ghost) >> utilities::PetscUtilities::checkError;
    DMPlexGetSupportSize(dm, face, &nsupp) >> utilities::PetscUtilities::checkError;
    DMPlexGetTreeChildren(dm, face, &nchild, nullptr) >> utilities::PetscUtilities::checkError;
    if (ghost >= 0 || nsupp > 2 || nchild > 0) continue;

    // determine where to add the cell flux values
    const PetscInt* faceCells;
    PetscFVCellGeom *cgL, *cgR;
    DMPlexGetSupport(subDomain.GetDM(), face, &faceCells) >> utilities::PetscUtilities::checkError;
    DMPlexPointLocalRead(cellDM, faceCells[0], cellGeomArray, &cgL) >> utilities::PetscUtilities::checkError;
    DMPlexPointLocalRead(cellDM, faceCells[1], cellGeomArray, &cgR) >> utilities::PetscUtilities::checkError;


    PetscScalar *fL = nullptr, *fR = nullptr;
    const PetscScalar *liquidDensityL = nullptr, *liquidDensityR = nullptr;
    const PetscScalar *phiL = nullptr, *phiR = nullptr;
    const PetscScalar *densityVF_L = nullptr, *densityVF_R = nullptr;
    PetscInt cellLabelValue = regionValue;
    DMLabelGetValue(ghostLabel, faceCells[0], &ghost) >> utilities::PetscUtilities::checkError;
    if (regionLabel) {
        DMLabelGetValue(regionLabel, faceCells[0], &cellLabelValue) >> utilities::PetscUtilities::checkError;
    }
    if (ghost <= 0 && regionValue == cellLabelValue) {
        xDMPlexPointLocalRef(dm, faceCells[0], densityVFField.id, locFArray, &fL) >> utilities::PetscUtilities::checkError;
        xDMPlexPointLocalRead(auxDM, faceCells[0], liquidDensityField.id, auxArray, &liquidDensityL) >> utilities::PetscUtilities::checkError;
        xDMPlexPointLocalRead(dm, faceCells[0], phiField.id, locXArray, &phiL) >> utilities::PetscUtilities::checkError;
        xDMPlexPointLocalRead(dm, faceCells[0], densityVFField.id, locXArray, &densityVF_L) >> utilities::PetscUtilities::checkError;

    }

    cellLabelValue = regionValue;
    DMLabelGetValue(ghostLabel, faceCells[1], &ghost) >> utilities::PetscUtilities::checkError;
    if (regionLabel) {
        DMLabelGetValue(regionLabel, faceCells[1], &cellLabelValue) >> utilities::PetscUtilities::checkError;
    }
    if (ghost <= 0 && regionValue == cellLabelValue) {
        xDMPlexPointLocalRef(dm, faceCells[1], densityVFField.id, locFArray, &fR) >> utilities::PetscUtilities::checkError;
        xDMPlexPointLocalRead(auxDM, faceCells[1], liquidDensityField.id, auxArray, &liquidDensityR) >> utilities::PetscUtilities::checkError;
        xDMPlexPointLocalRead(dm, faceCells[1], phiField.id, locXArray, &phiR) >> utilities::PetscUtilities::checkError;
        xDMPlexPointLocalRead(dm, faceCells[1], densityVFField.id, locXArray, &densityVF_R) >> utilities::PetscUtilities::checkError;
    }


    // No cells to add the flux to. Skip
//    if (!fL && !fR) continue;


//    PetscInt nCutCells = 0;
//    nCutCells += (phiL && *phiL > phiRange[0] && *phiL < phiRange[1]);
//    nCutCells += (phiR && *phiR > phiRange[0] && *phiR < phiRange[1]);
//    if (nCutCells==0) continue; // Need at least one cut-cell, otherwise skip


    PetscFVFaceGeom* fg;
    DMPlexPointLocalRead(faceDM, face, faceGeomArray, &fg) >> utilities::PetscUtilities::checkError;

#if 0
    PetscReal facePhi = 0.0, faceGrad[3] = {0.0, 0.0, 0.0};

//    if (phiL && (*phiL < phiRange[0] || *phiL > phiRange[1])) fL = nullptr;
//    if (phiR && (*phiR < phiRange[0] || *phiR > phiRange[1])) fR = nullptr;




//    for (PetscInt c = 0; c < stencil.stencilSize; c++) {
//      PetscInt cell = stencil.stencil[c];
//      const PetscReal *cellPhi;
//      xDMPlexPointLocalRead(dm, cell, phiField.id, locXArray, &cellPhi) >> utilities::PetscUtilities::checkError;
//      facePhi += (stencil.weights[c])*(*cellPhi);

//      for (PetscInt d = 0; d < dim; ++d) {
//        faceGrad[d] += (stencil.gradientWeights[c*dim + d])*(*cellPhi);
//      }
//    }

    faceGaussianConv->Evaluate(face, nullptr, dm, phiField.id, locXArray, 0, 1, &facePhi);
    for (PetscInt d = 0; d < dim; ++d) {
      PetscInt dx[3] = {0, 0, 0};
      dx[d] = 1;
      faceGaussianConv->Evaluate(face, dx, dm, phiField.id, locXArray, 0, 1, &faceGrad[d]);
    }

    const PetscReal norm = ablate::utilities::MathUtilities::MagVector(dim, faceGrad);
    const PetscReal fac = Gamma*(epsilon - facePhi*(1.0 - facePhi)/norm);


//    fprintf(f1, "%+e\t%+e\t%+e\t", fg->centroid[0], fg->centroid[1], facePhi);

//    const PetscReal nrm = ablate::utilities::MathUtilities::MagVector(dim, fg->centroid);

    for (PetscInt d = 0; d < dim; ++d) {
      PetscReal flux = fac*faceGrad[d]*fg->normal[d];


      if (fL) fL[d] -= flux / cgL->volume;
      if (fR) fR[d] += flux / cgR->volume;


//      if (fL) fL[d] -= (*liquidDensityL) * flux / cgL->volume;
//      if (fR) fR[d] += (*liquidDensityR) * flux / cgR->volume;

//      fprintf(f1, "%+e\t", fac*faceGrad[d]);
//      flux *= fg->normal[d];
//      if (fL) fL[d] -= flux / cgL->volume;
//      if (fR) fR[d] += flux / cgR->volume;

    }
//    fprintf(f1,"\n");

#endif


PetscReal vel[2] = {100, 0};


for (PetscInt d = 0; d < dim; ++d) {
  PetscReal flux = vel[d]*fg->normal[d];

  if (fL) fL[d] -= *densityVF_L * flux / cgL->volume;
  if (fR) fR[d] += *densityVF_R * flux / cgR->volume;
}

  }
//  fclose(f1);

  solver.RestoreRange(faceRange);

  // clean up
  VecRestoreArrayRead(cellGeomVec, &cellGeomArray) >> utilities::PetscUtilities::checkError;
  VecRestoreArrayRead(faceGeomVec, &faceGeomArray) >> utilities::PetscUtilities::checkError;
  VecRestoreArrayRead(locX, &locXArray) >> utilities::PetscUtilities::checkError;
  VecRestoreArray(locFVec, &locFArray) >> utilities::PetscUtilities::checkError;
  VecRestoreArrayRead(auxVec, &auxArray) >> utilities::PetscUtilities::checkError;

//  printf("%s::%s::%d\n", __FILE__, __FUNCTION__, __LINE__);
//  exit(0);

  PetscFunctionReturn(0);

}


#include "registrar.hpp"
REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::IntSharp, "calculates interface sharpening flux",
         ARG(PetscReal, "Gamma", "Gamma, interface sharpening strenghth parameter"),
         ARG(PetscReal, "epsilon", "epsilon, interface thickness scale parameter. Will be a multiple of the smallest grid spacing.")
);
