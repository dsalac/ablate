#include "intSharp.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "finiteVolume/processes/twoPhaseEulerAdvection.hpp"
#include "utilities/petscSupport.hpp"
#include "utilities/petscUtilities.hpp"


void ablate::finiteVolume::processes::IntSharp::ClearData() {
  if (cellGaussianConv) cellGaussianConv->~GaussianConvolution();
  if (subDM) DMDestroy(&subDM);
  if (subIS) ISDestroy(&subIS);
}

ablate::finiteVolume::processes::IntSharp::~IntSharp() {
  ablate::finiteVolume::processes::IntSharp::ClearData();
}



void ablate::finiteVolume::processes::IntSharp::GetFieldVectors(const ablate::domain::SubDomain& subDomain, Vec *subLocalVec, Vec *subGlobalVec) {

    // A copy is made so that the VOF values can be compared when updating the other fields

    const ablate::domain::Field &vofField = subDomain.GetField(ablate::finiteVolume::processes::TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD);
    Vec entireVec = subDomain.GetVec(vofField);

    DMGetLocalVector(subDM, subLocalVec) >> ablate::utilities::PetscUtilities::checkError;
    DMGetGlobalVector(subDM, subGlobalVec) >> ablate::utilities::PetscUtilities::checkError;

    if (vofField.location == ablate::domain::FieldLocation::SOL) {

      // Copy the data to the global vec
      VecISCopy(entireVec, subIS, SCATTER_REVERSE, *subGlobalVec) >> ablate::utilities::PetscUtilities::checkError;

      // Populate the local vector
      DMGlobalToLocal(subDM, *subGlobalVec, INSERT_VALUES, *subLocalVec) >> ablate::utilities::PetscUtilities::checkError;

    } else if (vofField.location == ablate::domain::FieldLocation::AUX) {

      // Copy the data to the local vec
      VecISCopy(entireVec, subIS, SCATTER_REVERSE, *subLocalVec) >> ablate::utilities::PetscUtilities::checkError;

      // Populate the global vector
      DMLocalToGlobal(subDM, *subLocalVec, INSERT_VALUES, *subGlobalVec) >> ablate::utilities::PetscUtilities::checkError;
    } else {
      throw std::invalid_argument("Volume fraction field is not contained in either the SOL or AUX vecs!");
    }

}

// Every time the mesh changes
void ablate::finiteVolume::processes::IntSharp::Initialize(ablate::finiteVolume::FiniteVolumeSolver &flow) {


  ablate::domain::SubDomain& subDomain = flow.GetSubDomain();

  ClearData();

  // Get the DM that contains JUST the vof field. This will be duplicated so that we can use DMGetLocalVector, which is
  // orders-or-magnitude faster than anything else for repeated calls.
  const ablate::domain::Field vofField = subDomain.GetField(ablate::finiteVolume::processes::TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD);
  DM entireDM = subDomain.GetFieldDM(vofField);
  DMCreateSubDM(entireDM, 1, &vofField.id, &subIS, &subDM) >> ablate::utilities::PetscUtilities::checkError;

  // The geometry vectors must be attached to the DM before DMPlexPointGeometricData is called.
  //  It this isn't done the code may hang when run in parallel as DMPlexComputeGeometryFVM must be run by all ranks.
  //  When using DMPlexPointGeometricData not all ranks may reach it and it will hang.
  Vec cellGeomVec, faceGeomVec;
  DMPlexComputeGeometryFVM(subDM, &cellGeomVec, &faceGeomVec) >> ablate::utilities::PetscUtilities::checkError;
  PetscObjectCompose((PetscObject)subDM, "DMPlex_cellgeom_fvm", (PetscObject)cellGeomVec) >> ablate::utilities::PetscUtilities::checkError;
  PetscObjectCompose((PetscObject)subDM, "DMPlex_facegeom_fvm", (PetscObject)faceGeomVec) >> ablate::utilities::PetscUtilities::checkError;
  VecDestroy(&cellGeomVec) >> ablate::utilities::PetscUtilities::checkError;
  VecDestroy(&faceGeomVec) >> ablate::utilities::PetscUtilities::checkError;

  if (cellRange.is) flow.RestoreRange(cellRange);
  flow.GetCellRangeWithoutGhost(cellRange);

  reverseCellRange = ablate::domain::ReverseRange(cellRange);

  // Using cell-center data compute the gaussian convolution at a cell
  PetscInt dim = subDomain.GetDimensions();
  cellGaussianConv = std::make_shared<ablate::finiteVolume::stencil::GaussianConvolution>(subDM, 1, dim, dim);

}

ablate::finiteVolume::processes::IntSharp::IntSharp(const PetscReal epsilonFac) : epsilonFac(epsilonFac==0 ? 1.0 : epsilonFac) {}

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


  ablate::domain::SubDomain& subDomain = flow.GetSubDomain();
  PetscInt i = 0;
  for (auto fieldName : fieldList) {
    if (!(subDomain.ContainsField(fieldName))) {
      throw std::runtime_error("ablate::finiteVolume::processes::IntSharp expects a "+ fieldName +" field to be defined.");
    }
    const ablate::domain::Field field = subDomain.GetField(fieldName);
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



void UpdateSolVec(ablate::domain::SubDomain& subDomain, ablate::domain::Range cellRange, DM vofDM, Vec vofVec) {

  // Values in SOL
  const ablate::domain::Field &vofField = subDomain.GetField(ablate::finiteVolume::processes::TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD);
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

  const PetscScalar *vofArray;
  VecGetArrayRead(vofVec, &vofArray) >> ablate::utilities::PetscUtilities::checkError;

  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
    PetscInt cell = cellRange.GetPoint(c);

    const PetscReal *newPhi;
    PetscReal *oldPhi;
    DMPlexPointLocalRead(vofDM, cell, vofArray, &newPhi) >> ablate::utilities::PetscUtilities::checkError;
    xDMPlexPointLocalRef(solDM, cell, vofField.id, solArray, &oldPhi) >> ablate::utilities::PetscUtilities::checkError;

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
      density = euler[ablate::finiteVolume::CompressibleFlowFields::RHO] = (*densityG)*(*newPhi) + (*densityL)*(1 - *newPhi);
      euler[ablate::finiteVolume::CompressibleFlowFields::RHOE] = (*densityG)*(*newPhi)*(*internalEnergyG) + (*densityL)*(1 - *newPhi)*(*internalEnergyL) + ke;
      for (PetscInt d = 0; d < dim; d++) {
        euler[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] = density*velocity[d];
      }
    }
  }

  VecRestoreArray(solVec, &solArray) >> ablate::utilities::PetscUtilities::checkError;
  VecRestoreArrayRead(auxVec, &auxArray) >> ablate::utilities::PetscUtilities::checkError;
  VecRestoreArrayRead(vofVec, &vofArray) >> ablate::utilities::PetscUtilities::checkError;
}



PetscErrorCode ablate::finiteVolume::processes::IntSharp::IntSharpPreStep(TS flowTS, ablate::solver::Solver &solver) {
  PetscFunctionBegin;
  PetscFunctionReturn(ablate::finiteVolume::processes::IntSharp::IntSharpPreStage(flowTS, solver, 0.0));
}


/**************************************************/
//   A note about vector creation
//  For a 100x100 domain VecDuplicate is 91X faster than DMCreateLocalVector.
//  DMGetLocalVector (after the initial vector creation) is 1600X faster than
//  DMCreateLocalVector and 20X faster than VecDuplicate
//
//  For a 200x200 domain VecDuplicate is 133X faster than DMCreateLocalVector.
//  DMGetLocalVector (after the initial vector creation) is 4300X faster than
//  DMCreateLocalVector and 33X faster than VecDuplicate


PetscErrorCode ablate::finiteVolume::processes::IntSharp::IntSharpPreStage(TS flowTS, ablate::solver::Solver &solver, PetscReal stagetime) {
  PetscFunctionBegin;

  auto &fvSolver = dynamic_cast<ablate::finiteVolume::FiniteVolumeSolver &>(solver);

  ablate::domain::SubDomain& subDomain = fvSolver.GetSubDomain();

  ablate::domain::Range cellRange;
  fvSolver.GetCellRangeWithoutGhost(cellRange);

  // Get the VOF data.
  const ablate::domain::Field vofField = subDomain.GetField(ablate::finiteVolume::processes::TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD);
  DM vofDM = subDM;
  Vec vofVecs[2] = {nullptr, nullptr};
  PetscScalar *vofArrays[2] = {nullptr, nullptr};
  GetFieldVectors(subDomain, &vofVecs[LOCAL], &vofVecs[GLOBAL]);
  PetscCall(VecGetArray(vofVecs[LOCAL], &vofArrays[LOCAL]));
  PetscCall(VecGetArray(vofVecs[GLOBAL], &vofArrays[GLOBAL]));

  const PetscInt dim = subDomain.GetDimensions();

  PetscReal h;
  PetscCall(DMPlexGetMinRadius(vofDM, &h));
  h *= 2.0; // Min radius returns the distance between a cell-center and a face. Double it to get the average cell sizes
  const PetscReal vofRange[2] = {1e-8, 1 - 1e-8};

  PetscReal dt = 0.25;

  // Work vector for the mask
  PetscInt *mask;

  PetscCall(DMGetWorkArray(vofDM, cellRange.end - cellRange.start, MPIU_INT, &mask));
  PetscCall(PetscArrayzero(mask, cellRange.end - cellRange.start));
  mask -= cellRange.start;
  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
    const PetscInt cell = cellRange.GetPoint(c);
    PetscReal *vof;
    PetscCall(DMPlexPointLocalRef(vofDM, cell, vofArrays[GLOBAL], &vof));

    if (*vof > vofRange[0] && *vof < vofRange[1]) {
      const PetscInt *cells;
      PetscInt nCells = cellGaussianConv->GetCellList(cell, &cells);

      for (PetscInt i = 0; i < nCells; ++i) {
        const PetscInt id = reverseCellRange.GetIndex(cells[i]);
        if (id < cellRange.end) mask[id] = 1;
      }
    }
  }



//  {
//    char fname[255];
//    sprintf(fname, "vof_%05d.txt", 0);
//    SaveCellData(vofDM, vofVecs[LOCAL], fname, -1, 1, cellRange);
//  }

  MPI_Comm comm = subDomain.GetComm();



  PetscInt iter = 0;
  PetscReal maxDiff = PETSC_MAX_REAL;
  while (iter < 200 && maxDiff > 1e-3) {
    ++iter;

    maxDiff = -1;
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {

      if (mask[c]==0) continue;

      const PetscInt cell = cellRange.GetPoint(c);
      PetscReal *vof;
      PetscCall(DMPlexPointLocalRef(vofDM, cell, vofArrays[GLOBAL], &vof));

//      PetscReal g[3];
//      DMPlexCellGradFromCell(vofDM, cell, vofVec, -1, 0, g);
//      PetscReal nrm = ablate::utilities::MathUtilities::MagVector(dim, g);


      PetscReal nrm = 0.0;
      for (PetscInt d = 0; d < dim; ++d) {
        PetscInt dx[3] = {0, 0, 0};
        dx[d] = 1;
        PetscReal g;
        cellGaussianConv->Evaluate(cell, dx, vofDM, -1, vofArrays[LOCAL], 0, 1, &g);

        nrm += g*g;
      }
      nrm = PetscSqrtReal(nrm);


      const PetscReal a = *vof;

      // This creates an almost-tanh like profile using an epsilon of 0.5 < eps < 1
      PetscReal eps = epsilonFac*1.5*tanh(a*(1-a)*10) + 0.01;
      PetscReal dv = dt*(eps*h*nrm - a*(1-a))*(1-2*a);

      *vof += dv;
      *vof = PetscMin(1.0, PetscMax(*vof, 0.0));

      maxDiff = PetscMax(maxDiff, PetscAbsReal(a - *vof));

    }

    PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &maxDiff, 1, MPIU_REAL, MPIU_MAX, comm));


    PetscCall(DMGlobalToLocal(vofDM, vofVecs[GLOBAL], INSERT_VALUES, vofVecs[LOCAL]));


//if ((iter)%10==0) {
//    char fname[255];
//    sprintf(fname, "vof_%05ld.txt", iter);
//    SaveCellData(vofDM, vofVecs[LOCAL], fname, -1, 1, cellRange);
//    PetscPrintf(PETSC_COMM_WORLD, "%ld\t%e\n", iter, maxDiff);
//}

  }

  mask += cellRange.start;
  PetscCall(DMRestoreWorkArray(vofDM, cellRange.end - cellRange.start, MPIU_INT, &mask));

  UpdateSolVec(subDomain, cellRange, vofDM, vofVecs[GLOBAL]);

  PetscCall(VecRestoreArray(vofVecs[LOCAL], &vofArrays[LOCAL]));
  PetscCall(VecRestoreArray(vofVecs[GLOBAL], &vofArrays[GLOBAL]));
  PetscCall(DMRestoreGlobalVector(subDM, &vofVecs[GLOBAL]));
  PetscCall(DMRestoreLocalVector(subDM, &vofVecs[LOCAL]));


//  printf("%s::%s::%d\n", __FILE__, __FUNCTION__, __LINE__);
//  exit(0);

  PetscFunctionReturn(0);
}




#include "registrar.hpp"
REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::IntSharp, "calculates interface sharpening flux",
         OPT(PetscReal, "epsilonFac", "Scaling factor for the interface sharpening strength. Default is 1.")
);
