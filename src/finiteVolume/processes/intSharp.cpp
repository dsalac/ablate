#include "intSharp.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "registrar.hpp"
#include "utilities/constants.hpp"
#include "utilities/mathUtilities.hpp"
#include "utilities/petscSupport.hpp"
#include "utilities/petscUtilities.hpp"

ablate::finiteVolume::processes::IntSharp::IntSharp(PetscReal Gamma, PetscReal epsilon) : Gamma(Gamma), epsilon(epsilon) {}

ablate::finiteVolume::processes::IntSharp::~IntSharp() { }

void ablate::finiteVolume::processes::IntSharp::Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) {
    flow.RegisterRHSFunction(ComputeTerm, this);
}



// Called every time the mesh changes
void ablate::finiteVolume::processes::IntSharp::Initialize(ablate::finiteVolume::FiniteVolumeSolver &solver) {
  IntSharp::subDomain = solver.GetSubDomainPtr();

}



void SaveVertexDataInt(DM dm, Vec vec, const char fname[255], const ablate::domain::Field *field, PetscInt Nc, std::shared_ptr<ablate::domain::SubDomain> subDomain) {

  ablate::domain::Range range;
  const PetscReal    *array, *val;
  PetscInt      dim = subDomain->GetDimensions();
  MPI_Comm      comm = PetscObjectComm((PetscObject)dm);
  int rank, size;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  ablate::domain::GetRange(dm, nullptr, 0, range);

  VecGetArrayRead(vec, &array) >> ablate::utilities::PetscUtilities::checkError;

  for (PetscInt r = 0; r < size; ++r) {
    if ( rank==r ) {

      FILE *f1;
      if ( rank==0 ) f1 = fopen(fname, "w");
      else f1 = fopen(fname, "a");

      for (PetscInt v = range.start; v < range.end; ++v) {
        PetscInt vert = range.points ? range.points[v] : v;
        PetscScalar *coords;

        DMPlexPointLocalFieldRead(dm, vert, field->id, array, &val) >> ablate::utilities::PetscUtilities::checkError;

        DMPlexVertexGetCoordinates(dm, 1, &vert, &coords);

        for (PetscInt d = 0; d < dim; ++d) {
          fprintf(f1, "%+.16e\t", coords[d]);
        }

        for (PetscInt i = 0; i < Nc; ++i) {
          fprintf(f1, "%+.16e\t", val[i]);
        }
        fprintf(f1, "\n");

        DMPlexVertexRestoreCoordinates(dm, 1, &vert, &coords);
      }

      fclose(f1);
    }
    MPI_Barrier(comm);
  }


  VecRestoreArrayRead(vec, &array) >> ablate::utilities::PetscUtilities::checkError;
  ablate::domain::RestoreRange(range);
}

PetscErrorCode ablate::finiteVolume::processes::IntSharp::ComputeTerm(const FiniteVolumeSolver &solver, DM dm, PetscReal time, Vec solVec, Vec fVec, void *ctx) {
    PetscFunctionBegin;

    //get fields
    ablate::finiteVolume::processes::IntSharp *process = (ablate::finiteVolume::processes::IntSharp *)ctx;
    std::shared_ptr<ablate::domain::SubDomain> subDomain = process->subDomain;
    const PetscInt dim = subDomain->GetDimensions();

    DM  auxDM = subDomain->GetAuxDM();
    Vec auxVec = subDomain->GetAuxVector();

    const auto &vofField = subDomain->GetField(TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD);
    const auto &eulerField = subDomain->GetField(ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD);

    const auto &vertexNormalField = subDomain->GetField("vertexNormal");

    DM eulerDM = subDomain->GetFieldDM(eulerField); // Get an euler-specific DM in case it's not in the same solution vector as the VOF field

    const auto vofID = vofField.id;
    const auto eulerID = eulerField.id;
    const auto vertexNormalID = vertexNormalField.id;

    // get vecs/arrays
    PetscScalar *auxArray, *fArray, *solArray;

    VecGetArray(auxVec, &auxArray);
    VecGetArray(fVec, &fArray);
    VecGetArray(solVec, &solArray);

    // get ranges
    ablate::domain::Range cellRange, vertRange;

    solver.GetCellRangeWithoutGhost(cellRange);
    subDomain->GetRange(nullptr, 0, vertRange);

    //march over vertices
    for (PetscInt v = vertRange.start; v < vertRange.end; v++) {

        const PetscInt vert = vertRange.GetPoint(v);

        PetscInt nCells, *cells;
        PetscScalar vof = 0.0;
        DMPlexVertexGetCells(dm, vert, &nCells, &cells);

        for (PetscInt c = 0; c < nCells; ++c) {
          const PetscScalar *cellVOF;
          xDMPlexPointLocalRead(dm, cells[c], vofID, solArray, &cellVOF) >> utilities::PetscUtilities::checkError;
          vof += *cellVOF;
        }
        vof /= nCells;
        DMPlexVertexRestoreCells(dm, vert, &nCells, &cells);

        PetscScalar *g;
        xDMPlexPointLocalRef(auxDM, vert, vertexNormalID, auxArray, &g) >> utilities::PetscUtilities::checkError;
        DMPlexVertexGradFromCell(dm, vert, solVec, vofID, 0, g) >> utilities::PetscUtilities::checkError;

        const PetscReal nrm = ablate::utilities::MathUtilities::MagVector(dim, g);

        if (nrm < PETSC_SMALL || vof < 0.001 || vof > 0.999) {
            for (PetscInt d = 0; d < dim; ++d) {
              g[d] = (process->Gamma)*((process->epsilon)*g[d]);
            }
        }
        else {
          for (PetscInt d = 0; d < dim; ++d) {
            g[d] = (process->Gamma)*((process->epsilon)*g[d] - vof*(1.0 - vof)*g[d]/nrm);
          }
        }
    }

    subDomain->UpdateAuxLocalVector();


    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {

        const PetscInt cell = cellRange.GetPoint(c);

        const PetscScalar *euler = nullptr;
        xDMPlexPointLocalRead(eulerDM, cell, eulerID, solArray, &euler) >> utilities::PetscUtilities::checkError;
        const PetscScalar density = euler[ablate::finiteVolume::CompressibleFlowFields::RHO];

        PetscScalar *eulerSource = nullptr;
        xDMPlexPointLocalRef(eulerDM, cell, eulerID, fArray, &eulerSource) >> utilities::PetscUtilities::checkError;

//        PetscScalar vel[dim];

        eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHO] = 0.0;
        eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOE] = 0.0;
        for (PetscInt d = 0; d < dim; ++d) {
            eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] = 0.0;
        }

        for (PetscInt d = 0; d < dim; ++d) {
//          vel[d] = euler[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density;

          PetscReal g[dim];
          DMPlexCellGradFromVertex(auxDM, cell, auxVec, vertexNormalID, d, g) >> utilities::PetscUtilities::checkError;
          eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHO] += g[d];
        }
        eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHO] *= density;  // This might be wrong

    }




    // cleanup
    VecRestoreArray(auxVec, &auxArray);
    VecRestoreArray(fVec, &fArray);
    VecRestoreArray(solVec, &solArray);

//    VecRestoreArray(auxvec, &auxArray);
    solver.RestoreRange(cellRange);
    subDomain->RestoreRange(vertRange);

    PetscFunctionReturn(0);

}

//PetscErrorCode ablate::finiteVolume::processes::IntSharp::ComputeTerm(const FiniteVolumeSolver &solver, DM dm, PetscReal time, Vec solVec, Vec fVec, void *ctx) {
//    PetscFunctionBegin;

//    //dm = sol DM
//    //locX = solvec
//    //locFVec = vector of conserved vars / eulerSource fields (rho, rhoe, rhov, ..., rhoet)
//    //auxvec = auxvec
//    //auxArray = auxArray
//    //notions of "process->" refer to the private variables: vertexDM, Gamma, epsilon. (the public variables are a subset: Gamma and epsilon)
//    //process->vertexDM = aux DM

//    //get fields
//    ablate::finiteVolume::processes::IntSharp *process = (ablate::finiteVolume::processes::IntSharp *)ctx;
//    std::shared_ptr<ablate::domain::SubDomain> subDomain = process->subDomain;
//    const PetscInt dim = subDomain->GetDimensions();

//    DM  auxDM = subDomain->GetAuxDM();
//    Vec auxVec = subDomain->GetAuxVector();

//    const auto &vofField = subDomain->GetField(TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD);
//    const auto &eulerField = subDomain->GetField(ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD);

//    const auto &vertexNormalField = subDomain->GetField("vertexNormal");
//    const auto &psiField = subDomain->GetField("curvature");

//    DM eulerDM = subDomain->GetFieldDM(eulerField); // Get an euler-specific DM in case it's not in the same solution vector as the VOF field

//    const auto vofID = vofField.id;
//    const auto eulerID = eulerField.id;
//    const auto vertexNormalID = vertexNormalField.id;
//    const auto psiID = psiField.id;

//    // get vecs/arrays
//    PetscScalar *auxArray, *fArray, *solArray;

//    VecGetArray(auxVec, &auxArray);
//    VecGetArray(fVec, &fArray);
//    VecGetArray(solVec, &solArray);

//    // get ranges
//    ablate::domain::Range cellRange, vertRange;

//    solver.GetCellRangeWithoutGhost(cellRange);
//    subDomain->GetRange(nullptr, 0, vertRange);

//    PetscReal h;
//    DMPlexGetMinRadius(dm, &h) >> ablate::utilities::PetscUtilities::checkError;
//    h *= 2.0; // Min radius returns the distance between a cell-center and a face. Double it to get the average cell size

//    const PetscReal varEps = 1e-100;
//    const PetscReal gamma = process->Gamma, eps = process->epsilon;
//    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
//      const PetscInt cell = cellRange.GetPoint(c);

//      const PetscScalar *vof;
//      PetscScalar *psi;

//      xDMPlexPointLocalRead(dm, cell, vofID, solArray, &vof) >> utilities::PetscUtilities::checkError;
//      xDMPlexPointLocalRef(auxDM, cell, psiID, auxArray, &psi) >> utilities::PetscUtilities::checkError;

////      if (*vof<0.0001) {
////        *psi = -20*h;
////      }
////      else if (*vof>0.9999) {
////        *psi = +20*h;
////      }
////      else {

//        *psi = eps * PetscLogScalar( (*vof + varEps)/(1 - *vof + varEps) );
////      }
//    }

//    subDomain->UpdateAuxLocalVector();


//    //march over vertices
//    for (PetscInt v = vertRange.start; v < vertRange.end; v++) {

//        const PetscInt vert = vertRange.GetPoint(v);

//        PetscInt nCells, *cells;
//        DMPlexVertexGetCells(auxDM, vert, &nCells, &cells) >> utilities::PetscUtilities::checkError;

//        PetscScalar vertPsi = 0.0, vertVOF = 0.0;
//        for (PetscInt c = 0; c < nCells; ++c) {
//          const PetscScalar *cellPsi;
//          xDMPlexPointLocalRead(auxDM, cells[c], psiID, auxArray, &cellPsi) >> utilities::PetscUtilities::checkError;
//          vertPsi += *cellPsi;

//          const PetscScalar *cellVOF;
//          xDMPlexPointLocalRead(dm, cells[c], vofID, solArray, &cellVOF) >> utilities::PetscUtilities::checkError;
//          vertVOF += *cellVOF;
//        }
//        vertPsi /= nCells;
//        vertVOF /= nCells;

//        DMPlexVertexRestoreCells(auxDM, vert, &nCells, &cells) >> utilities::PetscUtilities::checkError;

//        const PetscScalar tanhPSI = PetscTanhScalar(0.5*vertPsi/eps);


//        PetscReal gradPSI[dim];
//        DMPlexVertexGradFromCell(auxDM, vert, auxVec, psiID, 0, gradPSI) >> utilities::PetscUtilities::checkError;
//        const PetscReal gradPSInrm = ablate::utilities::MathUtilities::MagVector(dim, gradPSI);

//        PetscScalar *g;
//        xDMPlexPointLocalRead(auxDM, vert, vertexNormalID, auxArray, &g) >> utilities::PetscUtilities::checkError;
//        DMPlexVertexGradFromCell(dm, vert, solVec, vofID, 0, g) >> utilities::PetscUtilities::checkError;


//        if (vertVOF < 0.001 || vertVOF > 0.999) {
//          for (PetscInt d = 0; d < dim; ++d) {
//            g[d] = gamma*eps*g[d];
//          }
//        }
//        else {
//          for (PetscInt d = 0; d < dim; ++d) {
//            g[d] = gamma*(eps*g[d] - 0.25*(1.0 - PetscSqr(tanhPSI))*gradPSI[d]/gradPSInrm);
//          }
//        }


//    }

//    subDomain->UpdateAuxLocalVector();


//    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {

//        const PetscInt cell = cellRange.GetPoint(c);

//        const PetscScalar *euler = nullptr;
//        xDMPlexPointLocalRead(eulerDM, cell, eulerID, solArray, &euler) >> utilities::PetscUtilities::checkError;
//        const PetscScalar density = euler[ablate::finiteVolume::CompressibleFlowFields::RHO];

//        PetscScalar *eulerSource = nullptr;
//        xDMPlexPointLocalRef(eulerDM, cell, eulerID, fArray, &eulerSource) >> utilities::PetscUtilities::checkError;

////        PetscScalar vel[dim];

//        eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHO] = 0.0;
//        eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOE] = 0.0;
//        for (PetscInt d = 0; d < dim; ++d) {
//            eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] = 0.0;
//        }

//        for (PetscInt d = 0; d < dim; ++d) {
////          vel[d] = euler[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density;

//          PetscReal g[dim];
//          DMPlexCellGradFromVertex(auxDM, cell, auxVec, vertexNormalID, d, g) >> utilities::PetscUtilities::checkError;
//          eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHO] += g[d];
//        }
//        eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHO] *= density;  // This might be wrong

//    }

//    // cleanup
//    VecRestoreArray(auxVec, &auxArray);
//    VecRestoreArray(fVec, &fArray);
//    VecRestoreArray(solVec, &solArray);

//    solver.RestoreRange(cellRange);
//    subDomain->RestoreRange(vertRange);

//    PetscFunctionReturn(0);

//}

REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::IntSharp, "calculates interface regularization term",
         ARG(PetscReal, "Gamma", "Gamma, velocity scale parameter (approx. umax)"),
         ARG(PetscReal, "epsilon", "epsilon, interface thickness scale parameter (approx. h)")
);
