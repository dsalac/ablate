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



PetscErrorCode ablate::finiteVolume::processes::IntSharp::ComputeTerm(const FiniteVolumeSolver &solver, DM dm, PetscReal time, Vec solVec, Vec fVec, void *ctx) {
    PetscFunctionBegin;

    //dm = sol DM
    //locX = solvec
    //locFVec = vector of conserved vars / eulerSource fields (rho, rhoe, rhov, ..., rhoet)
    //auxvec = auxvec
    //auxArray = auxArray
    //notions of "process->" refer to the private variables: vertexDM, Gamma, epsilon. (the public variables are a subset: Gamma and epsilon)
    //process->vertexDM = aux DM

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

        PetscReal grad[dim];
        DMPlexVertexGradFromCell(auxDM, vert, auxVec, vofID, 0, grad) >> utilities::PetscUtilities::checkError;

        const PetscReal nrm = ablate::utilities::MathUtilities::MagVector(dim, grad);

        PetscScalar *g;
        xDMPlexPointLocalRef(auxDM, vert, vertexNormalID, auxArray, &g) >> utilities::PetscUtilities::checkError;
        DMPlexVertexGradFromCell(dm, vert, solVec, vofID, 0, g) >> utilities::PetscUtilities::checkError;

        const PetscScalar *vof;
        xDMPlexPointLocalRead(dm, vert, vofID, solArray, &vof) >> utilities::PetscUtilities::checkError;

        for (PetscInt d = 0; d < dim; ++d) {
          g[d] = (process->Gamma)*((process->epsilon)*g[d] - vof[0]*(1.0 - vof[0])*g[d]/nrm);
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

//    const PetscReal eps = 1e-100;
//    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
//      const PetscInt cell = cellRange.GetPoint(c);

//      const PetscScalar *vof;
//      PetscScalar *psi;

//      xDMPlexPointLocalRead(dm, cell, vofID, solArray, &vof) >> utilities::PetscUtilities::checkError;
//      xDMPlexPointLocalRef(auxDM, cell, psiID, auxArray, &psi) >> utilities::PetscUtilities::checkError;

//      *psi = eps * PetscLogScalar( (*vof + eps)/(1 - *vof + eps) );
//    }

//    subDomain->UpdateAuxLocalVector();


//    //march over vertices
//    for (PetscInt v = vertRange.start; v < vertRange.end; v++) {

//        const PetscInt vert = vertRange.GetPoint(v);

//        PetscReal gradPSI[dim];
//        DMPlexVertexGradFromCell(auxDM, vert, auxVec, psiID, 0, gradPSI) >> utilities::PetscUtilities::checkError;

//        ablate::utilities::MathUtilities::NormVector(dim, gradPSI, gradPSI);

//        PetscInt nCells, *cells;
//        DMPlexVertexGetCells(auxDM, vert, &nCells, &cells) >> utilities::PetscUtilities::checkError;

//        PetscScalar vertPsi = 0.0;
//        for (PetscInt c = 0; c < nCells; ++c) {
//          const PetscScalar *cellPsi;
//          xDMPlexPointLocalRead(auxDM, cells[c], psiID, auxArray, &cellPsi) >> utilities::PetscUtilities::checkError;
//          vertPsi += *cellPsi;
//        }
//        vertPsi /= nCells;

//        DMPlexVertexRestoreCells(auxDM, vert, &nCells, &cells) >> utilities::PetscUtilities::checkError;

//        const PetscScalar tanhPSI = PetscTanhScalar(0.5*vertPsi/(process->epsilon));

//        PetscScalar *g;
//        xDMPlexPointLocalRead(auxDM, vert, vertexNormalID, auxArray, &g) >> utilities::PetscUtilities::checkError;
//        DMPlexVertexGradFromCell(dm, vert, solVec, vofID, 0, g) >> utilities::PetscUtilities::checkError;

//        for (PetscInt d = 0; d < dim; ++d) {
//          g[d] = (process->Gamma)*((process->epsilon)*g[d] - 0.25*(1.0 - PetscSqr(tanhPSI))*gradPSI[d]);
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

////    VecRestoreArray(auxvec, &auxArray);
//    solver.RestoreRange(cellRange);
//    subDomain->RestoreRange(vertRange);

//    PetscFunctionReturn(0);

//}

REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::IntSharp, "calculates interface regularization term",
         ARG(PetscReal, "Gamma", "Gamma, velocity scale parameter (approx. umax)"),
         ARG(PetscReal, "epsilon", "epsilon, interface thickness scale parameter (approx. h)")
);
