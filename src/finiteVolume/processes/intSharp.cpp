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

  // Using cell-center data compute the gaussian convolution at a cell
  cellGaussianConv = std::make_shared<ablate::finiteVolume::stencil::GaussianConvolution>(dm, 1, dim, dim);

}

ablate::finiteVolume::processes::IntSharp::IntSharp(const PetscReal epsilonFac) : epsilonFac(epsilonFac) {}

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

  PetscReal dt = 0.25;

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

      // This creates an almost-tanh like profile using an epsilon of 0.5 < eps < 1
      PetscReal eps = epsilonFac*1.5*tanh(a*(1-a)*10) + 0.01;
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
         OPT(PetscReal, "epsilonFac", "Scaling factor for the interface sharpening strength. Default is 1.")
);
