#include "intSharp.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "finiteVolume/processes/twoPhaseEulerAdvection.hpp"
#include "utilities/constants.hpp"
#include "utilities/mathUtilities.hpp"
#include "utilities/petscSupport.hpp"
#include "utilities/petscUtilities.hpp"
#include "utilities/mpiUtilities.hpp"
#include "twoPhaseEulerAdvection.hpp"
#include "finiteVolume/stencils/faceStencilGenerator.hpp"
#include "finiteVolume/stencils/leastSquares.hpp"
#include "finiteVolume/stencils/leastSquaresAverage.hpp"
#include "finiteVolume/stencils/stencil.hpp"


void ablate::finiteVolume::processes::IntSharp::ClearData() {
  if (faceGaussianConv) faceGaussianConv->~GaussianConvolution();
  if (cellGaussianConv) cellGaussianConv->~GaussianConvolution();
}

ablate::finiteVolume::processes::IntSharp::~IntSharp() {
  ablate::finiteVolume::processes::IntSharp::ClearData();
}


// Every time the mesh changes
void ablate::finiteVolume::processes::IntSharp::Initialize(ablate::finiteVolume::FiniteVolumeSolver &flow) {


  ablate::domain::SubDomain& subDomain = flow.GetSubDomain();
  DM dm = subDomain.GetDM();

  PetscInt dim = subDomain.GetDimensions();


  // Using cell-center data compute the gaussian convolution at a cell, face, or vertex
  cellGaussianConv   = std::make_shared<ablate::finiteVolume::stencil::GaussianConvolution>(dm, 1, dim  , dim);
  faceGaussianConv   = std::make_shared<ablate::finiteVolume::stencil::GaussianConvolution>(dm, 1, dim-1, dim);
  vertexGaussianConv = std::make_shared<ablate::finiteVolume::stencil::GaussianConvolution>(dm, 1, 0    , dim);





  // Set up the gradient calculator
  std::unique_ptr<stencil::FaceStencilGenerator> faceStencilGenerator;
  if (dim == 1) {
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


//   Size up the stencil
  ablate::domain::Range faceRange;
  flow.GetFaceRange(faceRange);
  stencils.resize(faceRange.end - faceRange.start);

  // Compute the stencil for each face
  PetscInt iFace = 0;
  for (PetscInt f = faceRange.start; f < faceRange.end; ++f) {
    const PetscInt face = faceRange.GetPoint(f);

    auto& stencil = stencils[iFace++];

    // make sure that this is a valid face
    PetscInt ghost, nsupp, nchild;
    DMLabelGetValue(ghostLabel, face, &ghost) >> utilities::PetscUtilities::checkError;
    DMPlexGetSupportSize(dm, face, &nsupp) >> utilities::PetscUtilities::checkError;
    DMPlexGetTreeChildren(dm, face, &nchild, nullptr) >> utilities::PetscUtilities::checkError;
    if (ghost >= 0 || nsupp > 2 || nchild > 0) continue;

    faceStencilGenerator->Generate(face, stencil, subDomain, solverRegion, cellDM, cellGeomArray, faceDM, faceGeomArray);

  }

  flow.RestoreRange(faceRange);

  // clean up the geom
  VecRestoreArrayRead(cellGeomVec, &cellGeomArray) >> utilities::PetscUtilities::checkError;
  VecRestoreArrayRead(faceGeomVec, &faceGeomArray) >> utilities::PetscUtilities::checkError;

}

ablate::finiteVolume::processes::IntSharp::IntSharp(PetscReal Gamma, PetscReal epsilonIn) : Gamma(Gamma), epsilonIn(epsilonIn) {}

// Run once per simulation
void ablate::finiteVolume::processes::IntSharp::Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) {

  // List of required fields and locations
  std::string fieldList[] = { ablate::finiteVolume::CompressibleFlowFields::GASDENSITY_FIELD,
                              ablate::finiteVolume::CompressibleFlowFields::LIQUIDDENSITY_FIELD,
                              ablate::finiteVolume::CompressibleFlowFields::GASENERGY_FIELD,
                              ablate::finiteVolume::CompressibleFlowFields::LIQUIDENERGY_FIELD,
                              ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD,
                              ablate::finiteVolume::processes::TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD,
                              ablate::finiteVolume::processes::TwoPhaseEulerAdvection::DENSITY_VF_FIELD};

  ablate::domain::FieldLocation locationList[] = {
                              ablate::domain::FieldLocation::AUX,
                              ablate::domain::FieldLocation::AUX,
                              ablate::domain::FieldLocation::AUX,
                              ablate::domain::FieldLocation::AUX,
                              ablate::domain::FieldLocation::SOL,
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


//  flow.RegisterRHSFunction(IntSharpFlux,
//                         this,
//                         ablate::finiteVolume::processes::TwoPhaseEulerAdvection::DENSITY_VF_FIELD,
//                         {ablate::finiteVolume::processes::TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD, ablate::finiteVolume::processes::TwoPhaseEulerAdvection::DENSITY_VF_FIELD},
//                         {ablate::finiteVolume::CompressibleFlowFields::LIQUIDDENSITY_FIELD});


//  flow.RegisterRHSFunction(ComputeTerm, this);

    // Before each step, compute the source term over the entire dt
//  auto intSharpPreStage = std::bind(&ablate::finiteVolume::processes::IntSharp::IntSharpPreStage, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
//  flow.RegisterPreStage(intSharpPreStage);


  flow.RegisterPreStep([&](TS ts, ablate::solver::Solver &) { ablate::finiteVolume::processes::IntSharp::IntSharpPreStep(ts, flow); });


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

  std::shared_ptr<ablate::finiteVolume::stencil::GaussianConvolution> faceGaussianConv = process->faceGaussianConv;


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




static PetscInt cnt = 0;

void UpdateSolVec(ablate::domain::SubDomain& subDomain, ablate::domain::Range cellRange, DM vofDM, Vec vofVec, Vec vof0Vec) {

  // Values in SOL
  const ablate::domain::Field &eulerField = subDomain.GetField(ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD);
  const ablate::domain::Field &densityVFField = subDomain.GetField(ablate::finiteVolume::processes::TwoPhaseEulerAdvection::DENSITY_VF_FIELD);

  // Previously computed fields in AUX
  const ablate::domain::Field &gasDensityField = subDomain.GetField(ablate::finiteVolume::CompressibleFlowFields::GASDENSITY_FIELD);
  const ablate::domain::Field &liquidDensityField = subDomain.GetField(ablate::finiteVolume::CompressibleFlowFields::LIQUIDDENSITY_FIELD);
  const ablate::domain::Field &gasEnergyField = subDomain.GetField(ablate::finiteVolume::CompressibleFlowFields::GASENERGY_FIELD);
  const ablate::domain::Field &liquidEnergyField = subDomain.GetField(ablate::finiteVolume::CompressibleFlowFields::LIQUIDENERGY_FIELD);

  const PetscInt dim = subDomain.GetDimensions();

  Vec solVec = subDomain.GetSolutionVector();
  DM solDM = subDomain.GetDM();
  PetscScalar *solArray;
  VecGetArray(solVec, &solArray) >> ablate::utilities::PetscUtilities::checkError;

  Vec auxVec = subDomain.GetAuxVector();
  DM auxDM = subDomain.GetAuxDM();
  const PetscScalar *auxArray;
  VecGetArrayRead(auxVec, &auxArray) >> ablate::utilities::PetscUtilities::checkError;

  const PetscScalar *vofArray, *vof0Array;
  VecGetArrayRead(vofVec, &vofArray) >> ablate::utilities::PetscUtilities::checkError;
  VecGetArrayRead(vof0Vec, &vof0Array) >> ablate::utilities::PetscUtilities::checkError;

char fname[255];
sprintf(fname, "old_%ld.txt", cnt);
FILE *f1 = fopen(fname, "w");
sprintf(fname, "new_%ld.txt", cnt++);
FILE *f2 = fopen(fname, "w");

  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
    PetscInt cell = cellRange.GetPoint(c);

    const PetscReal *newPhi, *oldPhi;
    DMPlexPointLocalRead(vofDM, cell, vofArray, &newPhi) >> ablate::utilities::PetscUtilities::checkError;
    DMPlexPointLocalRead(vofDM, cell, vof0Array, &oldPhi) >> ablate::utilities::PetscUtilities::checkError;

PetscReal x[3];
DMPlexComputeCellGeometryFVM(solDM, cell, NULL, x, NULL);
{
  fprintf(f1, "%+e\t%+e\t%+e\t", x[0], x[1], *oldPhi);
  PetscReal *densityVF;
  xDMPlexPointLocalRead(solDM, cell, densityVFField.id, solArray, &densityVF) >> ablate::utilities::PetscUtilities::checkError;
  PetscReal *euler;
  xDMPlexPointLocalRead(solDM, cell, eulerField.id, solArray, &euler) >> ablate::utilities::PetscUtilities::checkError;
  fprintf(f1,"%+e\t%+e\t%+e\t%+e\t%+e\n", *densityVF, euler[0], euler[1], euler[2], euler[3]);
}

    if (PetscAbsReal(*newPhi - *oldPhi) > PETSC_SMALL) { // Only update those cells where the VOF has changed

      // Pre-computed values from twoPhaseEulerAdvection
      const PetscReal *densityG, *densityL;
      const PetscReal *internalEnergyG, *internalEnergyL;
      xDMPlexPointLocalRead(auxDM, cell, gasDensityField.id, auxArray, &densityG) >> ablate::utilities::PetscUtilities::checkError;
      xDMPlexPointLocalRead(auxDM, cell, liquidDensityField.id, auxArray, &densityL) >> ablate::utilities::PetscUtilities::checkError;
      xDMPlexPointLocalRead(auxDM, cell, gasEnergyField.id, auxArray, &internalEnergyG) >> ablate::utilities::PetscUtilities::checkError;
      xDMPlexPointLocalRead(auxDM, cell, liquidEnergyField.id, auxArray, &internalEnergyL) >> ablate::utilities::PetscUtilities::checkError;

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
      density = euler[ablate::finiteVolume::CompressibleFlowFields::RHO] = (*densityG)*(*newPhi) + (*densityL)*(1 - *newPhi);
      euler[ablate::finiteVolume::CompressibleFlowFields::RHOE] = (*densityG)*(*newPhi)*(*internalEnergyG) + (*densityL)*(1 - *newPhi)*(*internalEnergyL) + ke;
      for (PetscInt d = 0; d < dim; d++) {
        euler[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] = density*velocity[d];
      }
    }

{
  fprintf(f2, "%+e\t%+e\t%+e\t", x[0], x[1], *newPhi);
  PetscReal *densityVF;
  xDMPlexPointLocalRead(solDM, cell, densityVFField.id, solArray, &densityVF) >> ablate::utilities::PetscUtilities::checkError;
  PetscReal *euler;
  xDMPlexPointLocalRead(solDM, cell, eulerField.id, solArray, &euler) >> ablate::utilities::PetscUtilities::checkError;
  fprintf(f2,"%+e\t%+e\t%+e\t%+e\t%+e\n", *densityVF, euler[0], euler[1], euler[2], euler[3]);
}

  }

fclose(f1);
fclose(f2);

  VecRestoreArray(solVec, &solArray) >> ablate::utilities::PetscUtilities::checkError;
  VecRestoreArrayRead(auxVec, &auxArray) >> ablate::utilities::PetscUtilities::checkError;
  VecRestoreArrayRead(vofVec, &vofArray) >> ablate::utilities::PetscUtilities::checkError;
  VecRestoreArrayRead(vof0Vec, &vof0Array) >> ablate::utilities::PetscUtilities::checkError;
}



PetscErrorCode ablate::finiteVolume::processes::IntSharp::IntSharpPreStep(TS flowTS, ablate::solver::Solver &solver) {
  PetscFunctionBegin;
  PetscFunctionReturn(ablate::finiteVolume::processes::IntSharp::IntSharpPreStage(flowTS, solver, 0.0));
}


void EvaluateStencil(ablate::finiteVolume::stencil::Stencil stencil,  PetscFVFaceGeom* fg, DM dm, PetscInt dim, PetscInt id, PetscScalar *array, PetscReal *faceF, PetscReal *faceG) {
  *faceF = 0.0;
  for (PetscInt d = 0; d < dim; ++d) faceG[d] = 0.0;

  for (PetscInt c = 0; c < stencil.stencilSize; c++) {
    PetscInt cell = stencil.stencil[c];
    const PetscReal *f;
    xDMPlexPointLocalRead(dm, cell, id, array, &f) >> ablate::utilities::PetscUtilities::checkError;
    *faceF += (stencil.weights[c])*(*f);
    for (PetscInt d = 0; d < dim; ++d) {
      faceG[d] += (stencil.gradientWeights[c*dim + d])*(*f);
    }

//    *faceF += (stencil.weights[c])*(array[cell]);
//    for (PetscInt d = 0; d < dim; ++d) {
//      faceG[d] += (stencil.gradientWeights[c*dim + d])*(array[cell]);
//    }
  }

}

//PetscErrorCode ablate::finiteVolume::processes::IntSharp::IntSharpPreStage(TS flowTS, ablate::solver::Solver &solver, PetscReal stagetime) {
//  PetscFunctionBegin;

//  auto &fvSolver = dynamic_cast<ablate::finiteVolume::FiniteVolumeSolver &>(solver);

//  ablate::domain::SubDomain& subDomain = fvSolver.GetSubDomain();


//  // Get the VOF data.
//  const ablate::domain::Field vofField = subDomain.GetField(ablate::finiteVolume::processes::TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD);
//  Vec vofVec;
//  DM vofDM;
//  IS vofIS;
//  PetscScalar *vofArray;
//  subDomain.GetFieldLocalVector(vofField, stagetime, &vofIS, &vofVec, &vofDM) >> utilities::PetscUtilities::checkError;
//  VecGetArray(vofVec, &vofArray) >> utilities::PetscUtilities::checkError;


//  // check for ghost cells
//  DMLabel ghostLabel;
//  DMGetLabel(subDomain.GetDM(), "ghost", &ghostLabel) >> utilities::PetscUtilities::checkError;

//  // Get the solver region
//  const std::shared_ptr<ablate::domain::Region>& solverRegion = fvSolver.GetRegion();
//  DMLabel regionLabel;
//  PetscInt regionValue;
//  ablate::domain::Region::GetLabel(solverRegion, subDomain.GetDM(), regionLabel, regionValue);


//  // Used to update the VOF
//  Vec rhsVec;
//  PetscScalar *rhsArray;
//  VecDuplicate(vofVec, &rhsVec) >> utilities::PetscUtilities::checkError;
//  VecZeroEntries(rhsVec) >> utilities::PetscUtilities::checkError;
//  VecGetArray(rhsVec, &rhsArray) >> utilities::PetscUtilities::checkError;

//  // Store the original values. Needed to update conserved variables
//  Vec vof0Vec;
//  PetscScalar *vof0Array;
//  VecDuplicate(vofVec, &vof0Vec) >> utilities::PetscUtilities::checkError;
//  VecCopy(vofVec, vof0Vec) >> utilities::PetscUtilities::checkError;
//  VecGetArray(vof0Vec, &vof0Array) >> utilities::PetscUtilities::checkError;


//  const PetscReal vofRange[2] = {1e-8, 1 - 1e-8};


//  ablate::domain::Range faceRange, cellRange;
//  fvSolver.GetFaceRange(faceRange);
//  fvSolver.GetCellRangeWithoutGhost(cellRange);

//  const PetscInt dim = subDomain.GetDimensions();

//  PetscReal h;
//  PetscCall(DMPlexGetMinRadius(vofDM, &h));
//  h *= 2.0; // Min radius returns the distance between a cell-center and a face. Double it to get the average cell sizes

//  PetscReal dt = h;
//  Vec nrmVec;
//  VecDuplicate(vofVec, &nrmVec);
//  PetscScalar *nrmArray;
//  VecGetArray(nrmVec, &nrmArray);









//Vec gxVec, gyVec;
//PetscScalar *gxArray, *gyArray;
//VecDuplicate(vofVec, &gxVec);
//VecDuplicate(vofVec, &gyVec);
//VecGetArray(gxVec, &gxArray);
//VecGetArray(gyVec, &gyArray);



//printf("e=%.16e\n", epsilon);







//  for (PetscInt iter = 0; iter < 500; ++iter) {




//for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
//  const PetscInt cell = cellRange.GetPoint(c);
//  PetscReal g[3] = {0.0, 0.0, 0.0};
//  DMPlexCellGradFromCell(vofDM, cell, vofVec, -1, 0, g);

//  PetscReal *val;
//  DMPlexPointLocalRef(vofDM, cell, gxArray, &val) >> utilities::PetscUtilities::checkError;
//  *val = g[0];
//  DMPlexPointLocalRef(vofDM, cell, gyArray, &val) >> utilities::PetscUtilities::checkError;
//  *val = g[1];
//}

//char fname[244];
//sprintf(fname, "grad_%ld.txt", iter);
//FILE *f1 = fopen(fname, "w");
//for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
//  const PetscInt cell = cellRange.GetPoint(c);

//  PetscReal x[2];
//  DMPlexComputeCellGeometryFVM(vofDM, cell, NULL, x, NULL);
//  fprintf(f1, "%+e\t%+e\t", x[0], x[1]);

//  PetscReal *vof;
//  DMPlexPointLocalRef(vofDM, cell, vofArray, &vof) >> utilities::PetscUtilities::checkError;
//  fprintf(f1, "%+e\t", *vof);


//  PetscReal *val;
//  DMPlexPointLocalRef(vofDM, cell, gxArray, &val) >> utilities::PetscUtilities::checkError;
//  fprintf(f1, "%+e\t", *val);
//  DMPlexPointLocalRef(vofDM, cell, gyArray, &val) >> utilities::PetscUtilities::checkError;
//  fprintf(f1, "%+e\t", *val);


//  PetscReal lap = 0.0;
//  PetscReal g[3] = {0.0, 0.0, 0.0};
//  DMPlexCellGradFromCell(vofDM, cell, gxVec, -1, 0, g);
//  lap += g[0];
//  DMPlexCellGradFromCell(vofDM, cell, gyVec, -1, 0, g);
//  lap += g[1];
//  fprintf(f1, "%+e\n", lap);
//}
//fclose(f1);










//    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
//      const PetscInt cell = cellRange.GetPoint(c);
//      PetscReal *vof;
//      DMPlexPointLocalRef(vofDM, cell, vofArray, &vof) >> utilities::PetscUtilities::checkError;

//      if (*vof > vofRange[0] && *vof < vofRange[1]) {

//        PetscReal *vof0;
//        DMPlexPointLocalRef(vofDM, cell, vof0Array, &vof0) >> utilities::PetscUtilities::checkError;

//        PetscReal g[3] = {0.0, 0.0, 0.0};

////        for (PetscInt d = 0; d < dim; ++d) {
////          PetscInt dx[3] = {0, 0, 0};
////          dx[d] = 1;
////          cellGaussianConv->Evaluate(cell, dx, vofDM, -1, vofArray, 0, 1, &g[d]);
////        }

//        DMPlexCellGradFromCell(vofDM, cell, vofVec, -1, 0, g);

//        PetscReal nrm = ablate::utilities::MathUtilities::MagVector(dim, g);

//        PetscReal *dVof;
//        DMPlexPointLocalRef(vofDM, cell, rhsArray, &dVof) >> utilities::PetscUtilities::checkError;

//        PetscReal sgn = *vof0 - 0.5;
////        sgn = 8.0*sgn*sgn*sgn;
//        sgn *= 2.0;
//        *dVof = -Gamma*sgn*(nrm*epsilon - (*vof)*(1-*vof));
//      }

//    }



//    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
//      const PetscInt cell = cellRange.GetPoint(c);
//      PetscReal *vof;
//      DMPlexPointLocalRef(vofDM, cell, vofArray, &vof) >> utilities::PetscUtilities::checkError;

//      if (*vof > vofRange[0] && *vof < vofRange[1]) {
//        PetscReal *dVof;
//        DMPlexPointLocalRef(vofDM, cell, rhsArray, &dVof) >> utilities::PetscUtilities::checkError;
//        *vof += dt*(*dVof);
//        *vof = PetscMin(1.0, PetscMax(*vof, 0.0));
//      }
//    }





//  }

//  VecRestoreArray(vof0Vec, &vof0Array) >> utilities::PetscUtilities::checkError;
//  VecRestoreArray(vofVec, &vofArray) >> utilities::PetscUtilities::checkError;
//  VecRestoreArray(rhsVec, &rhsArray) >> utilities::PetscUtilities::checkError;
//  VecDestroy(&rhsVec) >> utilities::PetscUtilities::checkError;

//  UpdateSolVec(subDomain, cellRange, vofDM, vofVec, vof0Vec);
//  VecDestroy(&vof0Vec) >> utilities::PetscUtilities::checkError;

//  subDomain.RestoreFieldLocalVector(vofField, &vofIS, &vofVec, &vofDM) >> utilities::PetscUtilities::checkError;
//  printf("%s::%s::%d\n", __FILE__, __FUNCTION__, __LINE__);
//  exit(0);
//  PetscFunctionReturn(0);
//}

void ablate::finiteVolume::processes::IntSharp::MemoryHelper(const Vec baseVec, Vec *newVec, PetscScalar **newArray) {
  VecDuplicate(baseVec, newVec) >> utilities::PetscUtilities::checkError;
  VecZeroEntries(*newVec) >> utilities::PetscUtilities::checkError;
  VecGetArray(*newVec, newArray) >> utilities::PetscUtilities::checkError;
  vecList.push_back({*newVec, *newArray});
}

void ablate::finiteVolume::processes::IntSharp::MemoryHelper(const DM dm, PetscBool isLocalVec, Vec *newVec, PetscScalar **newArray) {

  if (isLocalVec) DMCreateLocalVector(dm, newVec);
  else DMCreateGlobalVector(dm, newVec);
  VecZeroEntries(*newVec) >> utilities::PetscUtilities::checkError;
  VecGetArray(*newVec, newArray) >> utilities::PetscUtilities::checkError;
  vecList.push_back({*newVec, *newArray});
}

void ablate::finiteVolume::processes::IntSharp::MemoryHelper() {
  for (struct vecData data : vecList) {
    VecRestoreArray(data.vec, &data.array) >> ablate::utilities::PetscUtilities::checkError;
    VecDestroy(&data.vec) >> ablate::utilities::PetscUtilities::checkError;
  }
  vecList.clear();

}

template<class T> void ignore( const T& ) { }
#include <signal.h>
PetscErrorCode ablate::finiteVolume::processes::IntSharp::IntSharpPreStage(TS flowTS, ablate::solver::Solver &solver, PetscReal stagetime) {
  PetscFunctionBegin;

  auto &fvSolver = dynamic_cast<ablate::finiteVolume::FiniteVolumeSolver &>(solver);

  ablate::domain::SubDomain& subDomain = fvSolver.GetSubDomain();

  // Get the VOF data.
  const ablate::domain::Field vofField = subDomain.GetField(ablate::finiteVolume::processes::TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD);
  Vec vofVec;
  DM vofDM;
  IS vofIS;
  PetscScalar *vofArray;
  subDomain.GetFieldLocalVector(vofField, stagetime, &vofIS, &vofVec, &vofDM) >> utilities::PetscUtilities::checkError;
  VecGetArray(vofVec, &vofArray) >> utilities::PetscUtilities::checkError;

  Vec globVofVec;
  PetscScalar *globVofArray;
  MemoryHelper(vofDM, PETSC_FALSE, &globVofVec, &globVofArray);

  DMLocalToGlobal(vofDM, vofVec, INSERT_VALUES, globVofVec) >> ablate::utilities::PetscUtilities::checkError;

  // check for ghost cells
  DMLabel ghostLabel;
  DMGetLabel(subDomain.GetDM(), "ghost", &ghostLabel) >> utilities::PetscUtilities::checkError;

  // Get the solver region
  const std::shared_ptr<ablate::domain::Region>& solverRegion = fvSolver.GetRegion();
  DMLabel regionLabel;
  PetscInt regionValue;
  ablate::domain::Region::GetLabel(solverRegion, subDomain.GetDM(), regionLabel, regionValue);

  // Store the original values. Needed to update conserved variables
  Vec vof0Vec;
  PetscScalar *vof0Array;
  MemoryHelper(vofVec, &vof0Vec, &vof0Array);



  ablate::domain::Range faceRange, cellRange;
  fvSolver.GetFaceRange(faceRange);
  fvSolver.GetCellRangeWithoutGhost(cellRange);

  const PetscInt dim = subDomain.GetDimensions();

  PetscReal h;
  PetscCall(DMPlexGetMinRadius(vofDM, &h));
  h *= 2.0; // Min radius returns the distance between a cell-center and a face. Double it to get the average cell sizes
//  const PetscReal vofRange[2] = {1e-8, 1 - 1e-8};

  PetscReal dt = 100*h;

  {
    char fname[255];
    sprintf(fname, "vof_%05d.txt", 0);
    SaveCellData(vofDM, vofVec, fname, -1, 1, cellRange);
  }

  MPI_Comm comm = subDomain.GetComm();


  for (PetscInt iter = 1; iter < 1000; ++iter) {


    PetscReal maxDiff = -1;
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
      const PetscInt cell = cellRange.GetPoint(c);
      PetscReal *vof;
      DMPlexPointLocalRef(vofDM, cell, globVofArray, &vof) >> utilities::PetscUtilities::checkError;

//      if (*vof < vofRange[0] || *vof > vofRange[1]) continue;

//      PetscReal g[3];
//      DMPlexCellGradFromCell(vofDM, cell, vofVec, -1, 0, g);
//      PetscReal nrm = ablate::utilities::MathUtilities::MagVector(dim, g);


      PetscReal nrm = 0.0;
      for (PetscInt d = 0; d < dim; ++d) {
        PetscInt dx[3] = {0, 0, 0};
        dx[d] = 1;
        PetscReal g;
        cellGaussianConv->Evaluate(cell, dx, vofDM, -1, vofArray, 0, 1, &g);

        nrm += g*g;
      }
      nrm = PetscSqrtReal(nrm);

      const PetscReal a = *vof;

      PetscReal eps = 1.5*tanh(a*(1-a)*10) + 0.01;
      PetscReal dv = dt*(a*(a-1)*(1-2*a) + eps*h*(1-2*a)*nrm);

      *vof += dv;
      *vof = PetscMin(1.0, PetscMax(*vof, 0.0));

      maxDiff = PetscMax(maxDiff, PetscAbsReal(a - *vof));

    }

    MPIU_Allreduce(MPI_IN_PLACE, &maxDiff, 1, MPIU_REAL, MPIU_MAX, comm) >> ablate::utilities::MpiUtilities::checkError;


    DMGlobalToLocal(vofDM, globVofVec, INSERT_VALUES, vofVec) >> ablate::utilities::PetscUtilities::checkError;


if ((iter)%10==0) {
    char fname[255];
    sprintf(fname, "vof_%05ld.txt", iter);
    SaveCellData(vofDM, vofVec, fname, -1, 1, cellRange);
    PetscPrintf(PETSC_COMM_WORLD, "%ld\t%e\n", iter, maxDiff);
}

  }


  UpdateSolVec(subDomain, cellRange, vofDM, vofVec, vof0Vec);


  MemoryHelper();

  VecRestoreArray(vofVec, &vofArray) >> utilities::PetscUtilities::checkError;
  subDomain.RestoreFieldLocalVector(vofField, &vofIS, &vofVec, &vofDM) >> utilities::PetscUtilities::checkError;
  printf("%s::%s::%d\n", __FILE__, __FUNCTION__, __LINE__);
  exit(0);
  PetscFunctionReturn(0);
}




#include "registrar.hpp"
REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::IntSharp, "calculates interface sharpening flux",
         ARG(PetscReal, "Gamma", "Gamma, interface sharpening strenghth parameter"),
         ARG(PetscReal, "epsilon", "epsilon, interface thickness scale parameter. Will be a multiple of the smallest grid spacing.")
);
