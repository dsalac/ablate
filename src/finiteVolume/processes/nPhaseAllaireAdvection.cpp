#include "nPhaseAllaireAdvection.hpp"

#include <utility>
// #include "eos/stiffenedGas.hpp"
#include "eos/kthStiffenedGas.hpp"
#include "eos/nPhase.hpp"
#include "finiteVolume/nPhaseFlowFields.hpp"
#include "flowProcess.hpp"
#include "domain/region.hpp"
#include "domain/subDomain.hpp"
#include "parameters/emptyParameters.hpp"
#include "utilities/petscSupport.hpp"
#include "utilities/mathUtilities.hpp"

#include "intSharp.hpp"

#include <signal.h>

#define xexit(S, ...) {PetscFPrintf(MPI_COMM_WORLD, stderr,                                     \
  "\x1b[1m(%s:%d, %s)\x1b[0m\n  \x1b[1m\x1b[90mexiting:\x1b[0m " S "\n",    \
  __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__); exit(0);}

static inline void NormVector(PetscInt dim, const PetscReal *in, PetscReal *out) {
    PetscReal mag = 0.0;
    for (PetscInt d = 0; d < dim; d++) {
        mag += in[d] * in[d];
    }
    mag = PetscSqrtReal(mag);
    for (PetscInt d = 0; d < dim; d++) {
        out[d] = in[d] / mag;
    }
}
static inline PetscReal MagVector(PetscInt dim, const PetscReal *in) {
    PetscReal mag = 0.0;
    for (PetscInt d = 0; d < dim; d++) {
        mag += in[d] * in[d];
    }
    return PetscSqrtReal(mag);
}



ablate::finiteVolume::processes::NPhaseAllaireAdvection::NPhaseAllaireAdvection(std::shared_ptr<eos::EOS> eosNPhase, const std::shared_ptr<parameters::Parameters> &parametersIn,
                                                                                std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorNStiff)
    : eosNPhase(std::move(eosNPhase)), fluxCalculatorNStiff(std::move(fluxCalculatorNStiff)) {
    auto parameters = ablate::parameters::EmptyParameters::Check(parametersIn);
    // check that eos is nPhase
    if (!this->eosNPhase) {
        throw std::invalid_argument("EOS cannot be null");
    }

    auto nPhaseEOS = std::dynamic_pointer_cast<eos::NPhase>(this->eosNPhase);
    if (!nPhaseEOS) {
        throw std::invalid_argument("EOS must be of type NPhase");
    }

    // populate component eoses
    std::size_t phases = nPhaseEOS->GetNumberOfPhases();
    eosk.resize(phases);

    for (std::size_t k=0; k<phases; k++) {
        auto phaseEOS = nPhaseEOS->GetEOSk(k);
        auto kthEOS = std::dynamic_pointer_cast<eos::KthStiffenedGas>(phaseEOS);
        if (!kthEOS) {
            throw std::invalid_argument("Each phase EOS must be of type KthStiffenedGas");
        }
        eosk[k] = kthEOS;
    }

    // If there is a flux calculator assumed advection
    if (this->fluxCalculatorNStiff) {
        // cfl
        timeStepData.cfl = parameters->Get<PetscReal>("cfl", 0.5);
    }

    // Zalesak test parameters
    zalesakTest = parameters->Get<bool>("zalesakTest", false);
    if (zalesakTest) {
    //     T_zalesak = parameters->Get<PetscReal>("T_zalesak", 1.0);
        PetscPrintf(MPI_COMM_WORLD, "[NPhaseAllaireAdvection] Zalesak test enabled\n");
    }

    //(MPI_COMM_WORLD, "end of constructor\n");
}


ablate::finiteVolume::processes::NPhaseAllaireAdvection::~NPhaseAllaireAdvection() {
    // Destructor implementation

}

void ablate::finiteVolume::processes::NPhaseAllaireAdvection::MultiphaseFlowPostEvaluate(TS flowTs, ablate::solver::Solver &solver) {

    auto alphaAccessor = solver.GetSubDomain().GetSolutionAccessor(ALPHAK_FIELD);
    auto alpharhoAccessor = solver.GetSubDomain().GetSolutionAccessor(ALPHAKRHOK_FIELD);
    auto allaireAccessor = solver.GetSubDomain().GetSolutionAccessor(ALLAIRE_FIELD);
//printf("%s::%d\n", __FILE__, __LINE__);
    // Cell range without ghosts
    ablate::domain::Range cellRange;
    solver.GetCellRangeWithoutGhost(cellRange);

    const PetscInt nPhases = alphaAccessor.GetField().numberComponents;
    const PetscInt dim = solver.GetSubDomain().GetDimensions();

    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        PetscInt cell = cellRange.GetPoint(c);

        // Get the euler and density field
        auto    alpha = alphaAccessor[cell];
        auto alphaRho = alpharhoAccessor[cell];
        auto  allaire = allaireAccessor[cell];

        // Only update if in the global vector
        if (alpha) {

          PetscReal alphaSum = 0;
          for (PetscInt k = 0; k < nPhases; k++) {
            alpha[k] = PetscMax(0, PetscMin(1, alpha[k]));
            alphaRho[k] = PetscMax(0, alphaRho[k]);
            alphaSum += alpha[k];
          }

          for (PetscInt k = 0; k < nPhases; k++) {
            alpha[k] /= alphaSum;
            alphaRho[k] /= alphaSum;
          }

          allaire[ablate::finiteVolume::NPhaseFlowFields::RHOE] /= alphaSum;
          for (PetscInt d = 0; d < dim; ++d) allaire[ablate::finiteVolume::NPhaseFlowFields::RHOU + d] /= alphaSum;
        }
    }

    // cleanup
    solver.RestoreRange(cellRange);

}

void ablate::finiteVolume::processes::NPhaseAllaireAdvection::Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) {

    ablate::domain::SubDomain& subDomain = flow.GetSubDomain();

    subDM = subDomain.GetDM();

    // Create the decoder based upon the eoses
    decoder = CreateNPhaseDecoder(subDomain.GetDimensions(), eosk);

//    flow.RegisterPreRHSFunction(PressurePreRHS, this); // Seems to make it unstable for long periods of time

    flow.RegisterPreRHSFunction(AijPreRHS, this);



    // Check whether viscosity exists
    PetscBool hasViscosity = PETSC_FALSE;
    PetscInt nPhases = eosk.size();
    mu.reserve(nPhases);
    zeroAlpha.reserve(nPhases);
    for (const auto& phaseEOS : eosk) {
        auto kthEOS = std::dynamic_pointer_cast<eos::KthStiffenedGas>(phaseEOS);
        mu.push_back(kthEOS->GetViscosity());
        hasViscosity = hasViscosity || (mu.back() > 0);
        if (hasViscosity && mu.back() <= 0) throw std::invalid_argument("A valid viscosity must be set for all fields.");

        zeroAlpha.push_back(PETSC_FALSE);
    }




//    flow.RegisterRHSFunction(NPhaseFlowComputeNPhaseContinuousFlux, this,
//        solFieldList,
//        solFieldList,
//        {ablate::finiteVolume::NPhaseFlowFields::PRESSURE, ablate::finiteVolume::NPhaseFlowFields::UI}); // Continuous advection: faceInterpolant: Doesn't seem to work anymore

#if 1
    flow.RegisterRHSFunction(NPhaseFlowComputeNPhaseFlux, this, solFieldList, solFieldList, {
      ablate::finiteVolume::NPhaseFlowFields::PRESSURE,
      ablate::finiteVolume::NPhaseFlowFields::UI,
      ablate::finiteVolume::NPhaseFlowFields::RHO,
      ablate::finiteVolume::NPhaseFlowFields::RHOK,
      ablate::finiteVolume::NPhaseFlowFields::SOSK
    }); // Conservative advection with discontinuous flux: cellInterpolant
    flow.RegisterRHSFunction(NPhaseFlowAlphakCorrection, this); // Remove alpha * div(u)
    if (hasViscosity) flow.RegisterRHSFunction(DiffusionFlux, this, {ALLAIRE_FIELD}, {ALPHAK_FIELD}, {ablate::finiteVolume::NPhaseFlowFields::UI});
#else
  PetscPrintf(PETSC_COMM_WORLD, "Turning off RHS in %s::%d\n", __FILE__, __LINE__);
#endif

    // After evaluation re-set the vof fields
    flow.RegisterPostEvaluate(MultiphaseFlowPostEvaluate);


    // Register Zalesak test as source term if enabled
    if (zalesakTest) {
        PetscPrintf(MPI_COMM_WORLD, "[NPhaseAllaireAdvection::Setup] About to register Zalesak test\n");
        PetscPrintf(MPI_COMM_WORLD, "[NPhaseAllaireAdvection::Setup] ALPHAK field offset: %d\n", subDomain.GetField(ALPHAK_FIELD).offset);
        PetscPrintf(MPI_COMM_WORLD, "[NPhaseAllaireAdvection::Setup] ALPHAKRHOK field offset: %d\n", subDomain.GetField(ALPHAKRHOK_FIELD).offset);
        PetscPrintf(MPI_COMM_WORLD, "[NPhaseAllaireAdvection::Setup] ALLAIRE_FIELD field offset: %d\n", subDomain.GetField(ALLAIRE_FIELD).offset);

        flow.RegisterRHSFunction(static_cast<ablate::finiteVolume::CellInterpolant::PointFunction>(ZalesakTestSourceTerm),
            this,
            {ALLAIRE_FIELD, ALPHAKRHOK_FIELD},    // Outputs
            {ALLAIRE_FIELD, ALPHAKRHOK_FIELD, ALPHAK_FIELD},  // Inputs
            {});
        PetscPrintf(MPI_COMM_WORLD, "[NPhaseAllaireAdvection::Setup] Zalesak test registered successfully\n");
    }

    flow.RegisterComputeTimeStepFunction(ComputeCflTimeStep, &timeStepData, "cfl");
    timeStepData.computeSpeedOfSound = eosNPhase->GetThermodynamicFunction(eos::ThermodynamicProperty::SpeedOfSound, subDomain.GetFields());

    // List of fields that could be in AUX that need to be computed before each time step
    std::string auxUpdateFieldList[] = { ablate::finiteVolume::NPhaseFlowFields::PRESSURE,
                                   ablate::finiteVolume::NPhaseFlowFields::UI,
                                   ablate::finiteVolume::NPhaseFlowFields::TK,
                                   ablate::finiteVolume::NPhaseFlowFields::RHO,
                                   ablate::finiteVolume::NPhaseFlowFields::RHOK,
                                   ablate::finiteVolume::NPhaseFlowFields::EPSILON,
                                   ablate::finiteVolume::NPhaseFlowFields::EPSILONK,
                                   ablate::finiteVolume::NPhaseFlowFields::SOSK,
                                   ablate::finiteVolume::NPhaseFlowFields::AIJ
                                };

    for (auto field : auxUpdateFieldList) {
      if (subDomain.ContainsField(field) && (subDomain.GetField(field).location == ablate::domain::FieldLocation::AUX)) auxUpdateFields.push_back(field);
    }

    if (auxUpdateFields.size() > 0) {
      flow.RegisterAuxFieldUpdate(
            UpdateAuxFieldsNPhase, this, auxUpdateFields, solFieldList);
    }


    gaussConv = std::make_shared<ablate::finiteVolume::stencil::GaussianConvolution>(subDM, 0.75, subDomain.GetDimensions(), subDomain.GetDimensions());



}

// Update the volume fraction, velocity, temperature, pressure fields, and gas density fields (if they exist).
PetscErrorCode ablate::finiteVolume::processes::NPhaseAllaireAdvection::UpdateAuxFieldsNPhase(PetscReal time, PetscInt dim, const PetscFVCellGeom *cellGeom, const PetscInt uOff[],
                                                                                                   const PetscScalar *conservedValues, const PetscInt aOff[], PetscScalar *auxField, void *ctx) {
    PetscFunctionBeginUser;

    if (!auxField) PetscFunctionReturn(0);

    auto nPhaseAllaireAdvection = (NPhaseAllaireAdvection *)ctx;
    const std::size_t nPhases = nPhaseAllaireAdvection->eosk.size();
    DM subDM = nPhaseAllaireAdvection->subDM;

    // For cell center, the norm is unity
    //  The normal velocity is not used, so it doesn't matter
    PetscReal norm[3];
    norm[0] = 1;
    norm[1] = 1;
    norm[2] = 1;

    PetscReal density = 1.0;
    PetscReal *densityk;
    PetscReal normalVelocity = 0.0;  // uniform velocity in cell
    PetscReal velocity[3] = {0.0, 0.0, 0.0};
    PetscReal internalEnergy = 0.0;
    PetscReal *internalEnergyk;
    PetscReal a = 0;
    PetscReal *ak;
    PetscReal *Mk;
    PetscReal p = 0.0;  // pressure equilibrium
    PetscReal *Tk;

    DMGetWorkArray(subDM, nPhases, MPIU_REAL, &densityk)  >> utilities::PetscUtilities::checkError;
    DMGetWorkArray(subDM, nPhases, MPIU_REAL, &internalEnergyk)  >> utilities::PetscUtilities::checkError;
    DMGetWorkArray(subDM, nPhases, MPIU_REAL, &ak)  >> utilities::PetscUtilities::checkError;
    DMGetWorkArray(subDM, nPhases, MPIU_REAL, &Mk)  >> utilities::PetscUtilities::checkError;
    DMGetWorkArray(subDM, nPhases, MPIU_REAL, &Tk)  >> utilities::PetscUtilities::checkError;

if (time==-12345) dim = -2;
    if (conservedValues) {
//        try {
            nPhaseAllaireAdvection->decoder->DecodeNPhaseAllaireState(subDM, cellGeom->centroid,
                dim, uOff, conservedValues, norm, &density, densityk, &normalVelocity, velocity, &internalEnergy, internalEnergyk, &a, ak, Mk, &p, Tk);

//        } catch (const std::exception& e) {
//            throw;
//        }

        for (PetscInt d = 0; d < dim; d++) {
            velocity[d] = conservedValues[uOff[ALLAIRE_OFFSET] + NPhaseFlowFields::RHOU + d] / density;
        }
    }

    auto fields = nPhaseAllaireAdvection->auxUpdateFields.data();

    for (std::size_t f = 0; f < nPhaseAllaireAdvection->auxUpdateFields.size(); ++f) {

        if (fields[f] == NPhaseFlowFields::UI) {
//          auxField[aOff[f] + 0] = 1;
//          auxField[aOff[f] + 1] = 0;
            for (PetscInt d = 0; d < dim; d++) {
                auxField[aOff[f] + d] = velocity[d];
            }
        }
        else if (fields[f] == NPhaseFlowFields::PRESSURE) {
            auxField[aOff[f]] = p;
        }
        else if (fields[f] == NPhaseFlowFields::TK) {
            for (std::size_t k = 0; k < nPhaseAllaireAdvection->eosk.size(); k++) {
                auxField[aOff[f] + k] = Tk[k];
            }
        }
        else if (fields[f] == NPhaseFlowFields::RHO) {
            auxField[aOff[f]] = density;
        }
        else if (fields[f] == NPhaseFlowFields::RHOK) {
            for (std::size_t k = 0; k < nPhaseAllaireAdvection->eosk.size(); k++) {
                auxField[aOff[f] + k] = densityk[k];
            }
        }
        else if (fields[f] == NPhaseFlowFields::EPSILON) {
            auxField[aOff[f]] = internalEnergy;
        }
        else if (fields[f] == NPhaseFlowFields::EPSILONK) {
            for (std::size_t k = 0; k < nPhaseAllaireAdvection->eosk.size(); k++) {
                auxField[aOff[f] + k] = internalEnergyk[k];
            }
        }
        else if (fields[f] == NPhaseFlowFields::SOSK) {
            for (std::size_t k = 0; k < nPhaseAllaireAdvection->eosk.size(); k++) {
                auxField[aOff[f] + k] = ak[k];
            }
        }
        else if (fields[f] == NPhaseFlowFields::AIJ) {
            // Populate Aij for unique pairs (i<j): Aij = alpha_i / (alpha_i + alpha_j)
            std::vector<PetscBool> zeroAlpha = nPhaseAllaireAdvection->zeroAlpha;
            for (std::size_t i = 0; i < nPhases; i++) {
                for (std::size_t j = 0; j < nPhases; j++) {
                    if (zeroAlpha[i] || zeroAlpha[j] || i == j) {
                        auxField[aOff[f] + i*nPhases + j] = 0;
                    }
                    else {
                      PetscReal value = NAN;
                      PetscReal sum = conservedValues[uOff[ALPHAK_OFFSET] + i] + conservedValues[uOff[ALPHAK_OFFSET] + j];
                      if (sum < PETSC_MACHINE_EPSILON) value = 0;
                      else if (conservedValues[uOff[ALPHAK_OFFSET] + i] < PETSC_MACHINE_EPSILON) value = 0;
                      else if (conservedValues[uOff[ALPHAK_OFFSET] + j] < PETSC_MACHINE_EPSILON) value = 1;
                      else value = conservedValues[uOff[ALPHAK_OFFSET] + i] / sum;
                      auxField[aOff[f] + i*nPhases + j] = value;
                    }
                }
            }
        }
    }

    DMRestoreWorkArray(subDM, nPhases, MPIU_REAL, &densityk)  >> utilities::PetscUtilities::checkError;
    DMRestoreWorkArray(subDM, nPhases, MPIU_REAL, &internalEnergyk)  >> utilities::PetscUtilities::checkError;
    DMRestoreWorkArray(subDM, nPhases, MPIU_REAL, &ak)  >> utilities::PetscUtilities::checkError;
    DMRestoreWorkArray(subDM, nPhases, MPIU_REAL, &Mk)  >> utilities::PetscUtilities::checkError;
    DMRestoreWorkArray(subDM, nPhases, MPIU_REAL, &Tk)  >> utilities::PetscUtilities::checkError;

    PetscFunctionReturn(0);
}



PetscErrorCode ablate::finiteVolume::processes::NPhaseAllaireAdvection::PressurePreRHS(FiniteVolumeSolver &fvSolver, TS ts, PetscReal time, bool initialStage, Vec locX, void *ctx) {
    PetscFunctionBegin;
    auto process = (NPhaseAllaireAdvection *)ctx;

    ablate::domain::Range cellRange;
    fvSolver.GetCellRangeWithoutGhost(cellRange);

    const PetscInt pId = fvSolver.GetSubDomain().GetField(NPhaseFlowFields::PRESSURE).id;

    DM auxDM = fvSolver.GetSubDomain().GetAuxDM();
    Vec auxVec = fvSolver.GetSubDomain().GetAuxVector();

    Vec newAuxVec;
    PetscCall(DMGetGlobalVector(auxDM, &newAuxVec));
    PetscCall(DMLocalToGlobal(auxDM, auxVec, INSERT_VALUES, newAuxVec));

    PetscScalar *newArray;
    PetscCall(VecGetArray(newAuxVec, &newArray));

    const PetscScalar *auxArray;
    PetscCall(VecGetArrayRead(auxVec, &auxArray));


    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
      const PetscInt cell = cellRange.GetPoint(c);
      PetscScalar *p;
      PetscCall(DMPlexPointGlobalFieldRef(auxDM, cell, pId, newArray, &p));
      if (p) process->gaussConv->Evaluate(cell, nullptr, auxDM, pId, auxArray, 0, 1, p);
    }
    PetscCall(VecRestoreArray(newAuxVec, &newArray));
    PetscCall(VecRestoreArrayRead(auxVec, &auxArray));
    PetscCall(DMGlobalToLocal(auxDM, newAuxVec, INSERT_VALUES, auxVec));
    PetscCall(DMRestoreGlobalVector(auxDM, &newAuxVec));

    fvSolver.RestoreRange(cellRange);

    PetscFunctionReturn(0);

}


PetscErrorCode ablate::finiteVolume::processes::NPhaseAllaireAdvection::AijPreRHS(FiniteVolumeSolver &fvSolver, TS ts, PetscReal time, bool initialStage, Vec locX, void *ctx) {
    PetscFunctionBegin;

    auto process = (NPhaseAllaireAdvection *)ctx;

    if (process->aijPreStageHasRun) PetscFunctionReturn(PETSC_SUCCESS);



    ablate::domain::Range cellRange;
    fvSolver.GetCellRangeWithoutGhost(cellRange);

    const PetscInt alphaId = fvSolver.GetSubDomain().GetField(NPhaseFlowFields::ALPHAK).id;
    const PetscInt aijId = fvSolver.GetSubDomain().GetField(NPhaseFlowFields::AIJ).id;

    DM auxDM = fvSolver.GetSubDomain().GetAuxDM();
    Vec auxVec = fvSolver.GetSubDomain().GetAuxVector();

    PetscScalar *auxArray;
    PetscCall(VecGetArray(auxVec, &auxArray));

    std::vector<PetscBool> zeroAlpha = process->zeroAlpha;
    std::size_t nPhases = zeroAlpha.size();

    PetscReal *alphaMax;
    PetscCall(DMGetWorkArray(auxDM, nPhases, MPIU_REAL, &alphaMax));
    for (std::size_t i = 0; i < nPhases; ++i) alphaMax[i] = PETSC_MIN_REAL;

    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
      const PetscInt cell = cellRange.GetPoint(c);
      const PetscScalar *vals;
      DMPlexPointLocalFieldRead(auxDM, cell, alphaId, auxArray, &vals);

      for (std::size_t i = 0; i < nPhases; ++i) {
        alphaMax[i] = PetscMax(alphaMax[i], vals[i]);
      }
    }

    PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, alphaMax, nPhases, MPIU_REAL, MPI_MAX, PetscObjectComm((PetscObject)auxDM)));

    for (std::size_t i = 0; i < nPhases; ++i) zeroAlpha[i] = (alphaMax[i] < PETSC_MACHINE_EPSILON);
    PetscCall(DMRestoreWorkArray(auxDM, nPhases, MPIU_REAL, &alphaMax));

    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
      const PetscInt cell = cellRange.GetPoint(c);

      PetscScalar *aij;
      DMPlexPointLocalFieldRef(auxDM, cell, aijId, auxArray, &aij);

      for (std::size_t i = 0; i < nPhases; ++i) {
        for (std::size_t j = 0; j < nPhases; ++j) {
          if (zeroAlpha[i] || zeroAlpha[j]) aij[i * nPhases + j] = 0;
        }
      }
    }

    PetscCall(VecRestoreArray(auxVec, &auxArray));

    fvSolver.RestoreRange(cellRange);

    process->aijPreStageHasRun = PETSC_TRUE;

    PetscFunctionReturn(0);

}

double ablate::finiteVolume::processes::NPhaseAllaireAdvection::ComputeCflTimeStep(TS ts, ablate::finiteVolume::FiniteVolumeSolver &flow, void *ctx) {
    // Get the dm and current solution vector
printf("%s::%d\n", __FILE__, __LINE__);
    exit(0);
    // (MPI_COMM_WORLD, "Computing CFL time step\n");
    DM dm;
    TSGetDM(ts, &dm) >> utilities::PetscUtilities::checkError;
    Vec v;
    TSGetSolution(ts, &v) >> utilities::PetscUtilities::checkError;

    // Get the flow param
    auto timeStepData = (TimeStepData *)ctx;

    // Get the fv geom
    PetscReal minCellRadius;
    DMPlexGetGeometryFVM(dm, NULL, NULL, &minCellRadius) >> utilities::PetscUtilities::checkError;

    // Get the valid cell range over this region
    ablate::domain::Range cellRange;
    flow.GetCellRange(cellRange);

    const PetscScalar *x;
    VecGetArrayRead(v, &x) >> utilities::PetscUtilities::checkError;

    // Get the dim from the dm
    PetscInt dim;
    DMGetDimension(dm, &dim) >> utilities::PetscUtilities::checkError;

    // assume the smallest cell is the limiting factor for now
    const PetscReal dx = 2.0 * minCellRadius;

    // Get field location for euler and densityYi
    auto allaireID = flow.GetSubDomain().GetField(ALLAIRE_FIELD).id;

    // March over each cell
    PetscReal dtMin = 1000.0;
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        PetscInt cell = cellRange.points ? cellRange.points[c] : c;

        const PetscReal *allaire;
        const PetscReal *conserved = NULL;
        DMPlexPointGlobalFieldRead(dm, cell, allaireID, x, &allaire) >> utilities::PetscUtilities::checkError;
        DMPlexPointGlobalRead(dm, cell, x, &conserved) >> utilities::PetscUtilities::checkError;

        if (allaire) {  // must be real cell and not ghost
            PetscReal rho = 998.23; //fix later; not using cfl compute for now
            // for (std::size_t k = 0; k < timeStepData->eosk.size(); k++) {
            //     rho += allaire[CompressibleFlowFields::ALPHAKRHOK + k];
            // }

            // Get the speed of sound from the eos
            PetscReal a;
            timeStepData->computeSpeedOfSound.function(conserved, &a, timeStepData->computeSpeedOfSound.context.get()) >> utilities::PetscUtilities::checkError;

            PetscReal velSum = 0.0;
            for (PetscInt d = 0; d < dim; d++) {
                velSum += PetscAbsReal(allaire[NPhaseFlowFields::RHOU + d]) / rho;
            }

            PetscReal dt = timeStepData->cfl * dx / (a + velSum);

            dtMin = PetscMin(dtMin, dt);
        }
    }
    VecRestoreArrayRead(v, &x) >> utilities::PetscUtilities::checkError;
    flow.RestoreRange(cellRange);
    return dtMin;
}

//FILE *f1 = fopen("flux.txt", "w");

PetscErrorCode ablate::finiteVolume::processes::NPhaseAllaireAdvection::NPhaseFlowComputeNPhaseFlux(PetscInt dim, const PetscFVFaceGeom *fg,
                                                                                                      const PetscInt *uOff,
                                                                                                      const PetscScalar *fieldL, const PetscScalar *fieldR,
                                                                                                      const PetscInt *aOff,
                                                                                                      const PetscScalar *auxL, const PetscScalar *auxR,
                                                                                                      PetscScalar *flux, void *ctx) {

  PetscFunctionBegin;

//++newCnt;
  auto nPhaseAllaireAdvection = (NPhaseAllaireAdvection *)ctx;
  std::size_t nPhases = nPhaseAllaireAdvection->eosk.size();
  DM dm = nPhaseAllaireAdvection->subDM;

  PetscReal norm[3];
  NormVector(dim, fg->normal, norm);
  const PetscReal areaMag = MagVector(dim, fg->normal);

  PetscScalar *ML_k = nullptr, *MR_k = nullptr;
  DMGetWorkArray(dm, nPhases, MPIU_REAL, &ML_k)  >> utilities::PetscUtilities::checkError;
  DMGetWorkArray(dm, nPhases, MPIU_REAL, &MR_k)  >> utilities::PetscUtilities::checkError;

  const PetscReal pL = auxL[aOff[0]];
  const PetscReal *velocityL = &auxL[aOff[1]];
  const PetscReal densityL = auxL[aOff[2]];
  const PetscReal *densityL_k = &auxL[aOff[3]];
  const PetscReal *aL_k = &auxL[aOff[4]];
  const PetscReal normalVelocityL = utilities::MathUtilities::DotVector(dim, velocityL, norm);



  const PetscReal pR = auxR[aOff[0]];
  const PetscReal *velocityR = &auxR[aOff[1]];
  const PetscReal densityR = auxR[aOff[2]];
  const PetscReal *densityR_k = &auxR[aOff[3]];
  const PetscReal *aR_k = &auxR[aOff[4]];
  const PetscReal normalVelocityR = utilities::MathUtilities::DotVector(dim, velocityR, norm);

  PetscReal aL = 0;
  PetscReal aR = 0;
  for (std::size_t k = 0; k < nPhases; ++k) {
    ML_k[k] = normalVelocityL / aL_k[k];
    MR_k[k] = normalVelocityR / aR_k[k];
    aL += fieldL[uOff[ALPHAK_OFFSET] + k] * densityL_k[k] * aL_k[k] * aL_k[k];
    aR += fieldR[uOff[ALPHAK_OFFSET] + k] * densityR_k[k] * aR_k[k] * aR_k[k];
  }
  aL = PetscSqrtReal(aL / densityL);
  aR = PetscSqrtReal(aR / densityR);

//      0: ablate::finiteVolume::NPhaseFlowFields::PRESSURE,
//      1: ablate::finiteVolume::NPhaseFlowFields::UI,
//      2: ablate::finiteVolume::NPhaseFlowFields::RHO,
//      3: ablate::finiteVolume::NPhaseFlowFields::RHOK,
//      4: ablate::finiteVolume::NPhaseFlowFields::SOSK


  PetscReal a12, m12, p12;

  nPhaseAllaireAdvection->fluxCalculatorNStiff->GetInterfaceValuesFunction()(
    nPhaseAllaireAdvection->fluxCalculatorNStiff->GetFluxCalculatorContext(),
    normalVelocityL, aL, densityL, pL,
    normalVelocityR, aR, densityR, pR,
    &a12, &m12, &p12);

//p12 = 115000;

  PetscReal vRiem = a12 * m12;

  const PetscScalar *alphak, *alphakRhok, *allaire;

  if (m12 > 0) { // Left
    alphak = &fieldL[uOff[ALPHAK_OFFSET]];
    alphakRhok = &fieldL[uOff[ALPHAKRHOK_OFFSET]];
    allaire = &fieldL[uOff[ALLAIRE_OFFSET]];
  }
  else { // Right
    alphak = &fieldR[uOff[ALPHAK_OFFSET]];
    alphakRhok = &fieldR[uOff[ALPHAKRHOK_OFFSET]];
    allaire = &fieldR[uOff[ALLAIRE_OFFSET]];
  }

//   solFieldList = {ALPHAK_FIELD, ALPHAKRHOK_FIELD, ALLAIRE_FIELD, VELDIV_FIELD};
  std::size_t offset = 0;

  // alpha
  for (std::size_t k = 0; k < nPhases; k++) flux[offset++] = vRiem * alphak[k] * areaMag;

  // rho*alpha
  for (std::size_t k = 0; k < nPhases; k++) flux[offset++] = vRiem * alphakRhok[k] * areaMag;

  // energy
  flux[offset++] = vRiem * (allaire[NPhaseFlowFields::RHOE] + p12) * areaMag;

  // momentum
  for (PetscInt d = 0; d < dim; d++) flux[offset++] = vRiem * allaire[NPhaseFlowFields::RHOU + d] * areaMag + p12 * fg->normal[d];

  // vel-div
//  PetscReal vel[2] = {sin(10*M_PI*fg->centroid[0]), cos(10*M_PI*fg->centroid[1])};
//  vRiem = utilities::MathUtilities::DotVector(dim, vel, norm);
  flux[offset++] = -vRiem * areaMag;

  for (std::size_t k = 0; k < offset; ++k) {
    if (PetscIsNanReal(flux[k])) throw std::runtime_error("A flux is NaN");
  }

  DMRestoreWorkArray(dm, nPhases, MPIU_REAL, &ML_k)  >> utilities::PetscUtilities::checkError;
  DMRestoreWorkArray(dm, nPhases, MPIU_REAL, &MR_k)  >> utilities::PetscUtilities::checkError;

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ablate::finiteVolume::processes::NPhaseAllaireAdvection::NPhaseFlowComputeNPhaseContinuousFlux(PetscInt dim, const PetscFVFaceGeom* fg,
  const PetscInt uOff[], const PetscInt uOff_x[],
  const PetscScalar fieldL[], const PetscScalar fieldR[], const PetscScalar field[], const PetscScalar grad[],
  const PetscInt aOff[], const PetscInt aOff_x[],
  const PetscScalar auxL[], const PetscScalar auxR[], const PetscScalar aux[], const PetscScalar gradAux[],
  PetscScalar flux[], void* ctx) {

  PetscFunctionBegin;

  auto nPhaseAllaireAdvection = (NPhaseAllaireAdvection *)ctx;
  std::size_t nPhases = nPhaseAllaireAdvection->eosk.size();

  const PetscReal p = aux[aOff[0]];
  const PetscReal *vel = &aux[aOff[1]];

  const PetscReal u_n = utilities::MathUtilities::DotVector(dim, vel, fg->normal);

  const PetscScalar *alphak, *alphakRhok, *allaire;

  if (u_n > 0) { // Left
    alphak = &fieldL[uOff[ALPHAK_OFFSET]];
    alphakRhok = &fieldL[uOff[ALPHAKRHOK_OFFSET]];
    allaire = &fieldL[uOff[ALLAIRE_OFFSET]];
  }
  else { // Right
    alphak = &fieldR[uOff[ALPHAK_OFFSET]];
    alphakRhok = &fieldR[uOff[ALPHAKRHOK_OFFSET]];
    allaire = &fieldR[uOff[ALLAIRE_OFFSET]];
  }

//   solFieldList = {ALPHAK_FIELD, ALPHAKRHOK_FIELD, ALLAIRE_FIELD};
  std::size_t offset = 0;

  // alpha
  for (std::size_t k = 0; k < nPhases; k++) flux[offset++] = u_n * alphak[k];

  // rho*alpha
  for (std::size_t k = 0; k < nPhases; k++) flux[offset++] = u_n * alphakRhok[k];

  // energy
  flux[offset++] = u_n * (allaire[NPhaseFlowFields::RHOE] + p);

  // momentum
  for (PetscInt d = 0; d < dim; d++) flux[offset++] = u_n * allaire[NPhaseFlowFields::RHOU + d] + p * fg->normal[d];

  // vel-div
  flux[offset++] = -u_n;

  for (std::size_t k = 0; k < offset; ++k) {
    if (PetscIsNanReal(flux[k])) throw std::runtime_error("A flux is NaN");
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}


/*

  The advection equation for alphaK is d(alphaK)/dt + u.grad(alphaK) = 0. In conservative form this is
    d(alphaK)/dt + div(alphaK * u) - alphaK * div(u) = 0. As everything gets shifted to the RHS we have
    d(alphaK)/dt = -div(alphaK * u) + alphaK * div(u),

  NPhaseFlowComputeNPhaseFlux handles the -div(alphaK * u) part. This function handles the + alphaK * div(u)
    part, with -div(u) being calculated in NPhaseFlowComputeNPhaseFlux. Note that all
    discontinuousFluxFunction-type calls return -div( stuff ).

  Note #1: According to "Generic five-equation model for compressible multi-material flows and its corresponding
    high-fidelity numerical algorithms" by He, Liu, and Li, the advection equation should be:
    d(alphaK)/dt + u.grad(alphaK) = alphaK*(lambdaK - 1)*div(u), where lambdaK accounts for different
    compressibility factors. This results in the following:

    d(alphaK)/dt + div(alphaK * u) - alphaK * div(u) = alphaK * (lambdaK - 1) * div(u)
    d(alphaK)/dt + div(alphaK * u) = alphaK * lambdaK * div(u)

    lambdaK = 1 assumes all materials have the same compressibility factor and this returns to the base Allaire model.

  Note #2: In the FVM the LHS is \int_cell dQ/dt. Assuming that dQ/dt is constant over a cell this results in
    V_{cell} dQ/dt. Thus, the RHS of the conservative form will be divided by V_{cell}, which is why the
    flux terms, when applied in cellInterpolant are divided by the cell volume and why the RHS contribution
    from this function do not need to be multiplied by the cell volume.
*/

PetscErrorCode ablate::finiteVolume::processes::NPhaseAllaireAdvection::NPhaseFlowAlphakCorrection(const FiniteVolumeSolver& flow, DM dm, PetscReal time, Vec locXVec, Vec locFVec, void* ctx) {
  PetscFunctionBegin;

//  auto process = (NPhaseAllaireAdvection *)ctx;
  ablate::domain::Range cellRange;
  flow.GetCellRangeWithoutGhost(cellRange);
  const PetscScalar *xArray;
  PetscScalar *fArray;
  PetscInt nPhases = flow.GetSubDomain().GetField(ALPHAK_FIELD).numberComponents;
  const PetscInt alphaId = flow.GetSubDomain().GetField(ALPHAK_FIELD).id;
  const PetscInt velDivId = flow.GetSubDomain().GetField("veldiv").id;

  PetscCall(VecGetArrayRead(locXVec, &xArray));
  PetscCall(VecGetArray(locFVec, &fArray));

  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
    const PetscInt cell = cellRange.GetPoint(c);

    const PetscScalar *alpha, *div;
    PetscScalar *alphaF;

//PetscReal x[2], vol;
//DMPlexPointGeometricData(dm, cell, &vol, x, NULL);
//if (PetscAbsReal(x[0] - 0.0567307) < 1e-6 && PetscAbsReal(x[1] - 0.0097922) < 1e-6) {
//  printf("Cell: %d\n", cell);
//  PetscInt nFaces;
//  DMPlexGetConeSize(dm, cell, &nFaces);
//  const PetscInt *faces;
//  DMPlexGetCone(dm, cell, &faces);

//  for (PetscInt i = 0; i < nFaces; ++i) {
//    PetscReal area, x[2], n[2];
//    DMPlexPointGeometricData(dm, faces[i], &area, x, n);
//    printf("quiver(%e,%e,%e,%e,0);\n", x[0], x[1], area*n[0], area*n[1]);
//    printf("n%d = [%+.16e %+.16e];\n", i+1, area*n[0], area*n[1]);

//    const PetscScalar *array;
//    PetscScalar       *coords = NULL;
//    PetscInt           numCoords;
//    PetscBool          isDG;
//    PetscCall(DMPlexGetCellCoordinates(dm, faces[i], &isDG, &numCoords, &array, &coords));
//    printf("plot([%e %e],[%e %e],'k');\n", coords[0], coords[2], coords[1], coords[3]);
//    PetscCall(DMPlexRestoreCellCoordinates(dm, faces[i], &isDG, &numCoords, &array, &coords));

//  }
//  printf("v=%.16e;\n", vol);
//  xexit("");

//}

    PetscCall(DMPlexPointLocalFieldRead(dm, cell, alphaId, xArray, &alpha));
    PetscCall(DMPlexPointLocalFieldRef(dm, cell, alphaId, fArray, &alphaF));
    PetscCall(DMPlexPointLocalFieldRead(dm, cell, velDivId, fArray, &div));

    for (PetscInt k = 0; k < nPhases; ++k) alphaF[k] += alpha[k] * div[0];

  }

  PetscCall(VecRestoreArrayRead(locXVec, &xArray));
  PetscCall(VecRestoreArray(locFVec, &fArray));
  flow.RestoreRange(cellRange);




  PetscFunctionReturn(PETSC_SUCCESS);

}


/* Modified from ablate::finiteVolume::processes::NavierStokesTransport */
PetscErrorCode ablate::finiteVolume::processes::NPhaseAllaireAdvection::DiffusionFlux(PetscInt dim, const PetscFVFaceGeom* fg,
  const PetscInt uOff[], const PetscInt uOff_x[],
  const PetscScalar fieldL[], const PetscScalar fieldR[], const PetscScalar field[], const PetscScalar grad[],
  const PetscInt aOff[], const PetscInt aOff_x[],
  const PetscScalar auxL[], const PetscScalar auxR[], const PetscScalar aux[], const PetscScalar gradAux[],
  PetscScalar flux[], void* ctx) {
    PetscFunctionBeginUser;

    auto process = (NPhaseAllaireAdvection *)ctx;

    // Compute the volume-averaged mixture viscosity
    PetscReal mu = 0;
    for (std::size_t k = 0; k < process->mu.size(); ++k) mu += field[uOff[0] + k] * process->mu[k];

    // Compute the stress tensor tau
    PetscReal tau[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};  // Maximum size without symmetry
    PetscCall(ComputeStressTensor(dim, mu, &gradAux[aOff_x[0]], tau));

    // for each velocity component
    for (PetscInt c = 0; c < dim; ++c) {
        PetscReal viscousFlux = 0.0;
        PetscReal  energyFlux = 0.0;

        // March over each direction
        for (PetscInt d = 0; d < dim; ++d) {
          viscousFlux += tau[c * dim + d] * fg->normal[d];    // tau[c][d].n[d]
           energyFlux += tau[c * dim + d] * aux[aOff[0] + d]; // tau[c][d].u[d]
        }

        // Add in the contribution. It's negative since it's on the RHS
        flux[NPhaseFlowFields::RHOE] -= energyFlux * fg->normal[c];
        flux[NPhaseFlowFields::RHOU + c] = -viscousFlux;

    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ablate::finiteVolume::processes::NPhaseAllaireAdvection::ComputeStressTensor(PetscInt dim, PetscReal mu, const PetscReal* gradVel, PetscReal* tau) {
    PetscFunctionBeginUser;
    // pre-compute the div of the velocity field
    PetscReal divVel = 0.0;
    for (PetscInt c = 0; c < dim; ++c) {
        divVel += gradVel[c * dim + c];
    }

    // March over each velocity component, u, v, w
    for (PetscInt c = 0; c < dim; ++c) {
        // March over each physical coordinates
        for (PetscInt d = 0; d < dim; ++d) {
            if (d == c) {
                // for the xx, yy, zz, components
                tau[c * dim + d] = 2.0 * mu * ((gradVel[c * dim + d]) - divVel / 3.0);
            } else {
                // for xy, xz, etc
                tau[c * dim + d] = mu * ((gradVel[c * dim + d]) + (gradVel[d * dim + c]));
            }
        }
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}


std::shared_ptr<ablate::finiteVolume::processes::NPhaseAllaireAdvection::NPhaseDecoder> ablate::finiteVolume::processes::NPhaseAllaireAdvection::CreateNPhaseDecoder(
    PetscInt dim, const std::vector<std::shared_ptr<eos::KthStiffenedGas>> &eosk) {


    // return std::make_shared<NStiffDecoder>(dim, eosk);
    std::vector<std::shared_ptr<ablate::eos::KthStiffenedGas>> stiffGases;

    for (const auto& eos : eosk) {
      auto stiffGas = std::dynamic_pointer_cast<ablate::eos::KthStiffenedGas>(eos);
      if (!stiffGas) {
        throw std::invalid_argument("All EOSs must be kthStiffenedGas for NPhaseAllaireAdvection");
      }
      stiffGases.push_back(stiffGas);
    }

    auto decoder = std::make_shared<NStiffDecoder>(dim, stiffGases); //this is where the error is

    return decoder;
}


ablate::finiteVolume::processes::NPhaseAllaireAdvection::NStiffDecoder::NStiffDecoder(PetscInt dim, const std::vector<std::shared_ptr<eos::KthStiffenedGas>> &eosk)
    : eosk(eosk) {
    //(MPI_COMM_WORLD, "Starting NStiffDecoder constructor\n");

    std::size_t phases = eosk.size();
    //(MPI_COMM_WORLD, "Input eosk size: %lu\n", phases);

    // Create the fake euler field
    //(MPI_COMM_WORLD, "Creating fake Allaire field\n");
    auto fakeAllaireField = ablate::domain::Field{.name = ALLAIRE_FIELD,
                                                .numberComponents = 1 + dim,
                                                .components = {},
                                                .id = PETSC_DEFAULT,
                                                .subId = PETSC_DEFAULT,
                                                .offset = 0,
                                                .location = ablate::domain::FieldLocation::SOL,
                                                .type = ablate::domain::FieldType::FVM,
                                                .tags = {}};

    // Initialize all vectors to the correct size first
    //(MPI_COMM_WORLD, "Initializing vectors\n");
    kAllaireFieldScratch.resize(phases);
    kComputeTemperature.resize(phases);
    kComputeInternalEnergy.resize(phases);
    kComputeSpeedOfSound.resize(phases);
    kComputePressure.resize(phases);

    // Now initialize each phase
    //(MPI_COMM_WORLD, "Initializing phase data\n");
    for (std::size_t k = 0; k < phases; k++) {
        if (!eosk[k]) {
            throw std::invalid_argument("EOS for phase " + std::to_string(k) + " is null");
        }
        //(MPI_COMM_WORLD, "Initializing phase %lu\n", k);
        kAllaireFieldScratch[k].resize(1 + dim);
        //(MPI_COMM_WORLD, "Getting thermodynamic functions for phase %lu\n", k);
        kComputeTemperature[k] = eosk[k]->GetThermodynamicFunction(eos::ThermodynamicProperty::Temperature, {fakeAllaireField});
        kComputeInternalEnergy[k] = eosk[k]->GetThermodynamicFunction(eos::ThermodynamicProperty::InternalSensibleEnergy, {fakeAllaireField});
        kComputeSpeedOfSound[k] = eosk[k]->GetThermodynamicFunction(eos::ThermodynamicProperty::SpeedOfSound, {fakeAllaireField});
        kComputePressure[k] = eosk[k]->GetThermodynamicFunction(eos::ThermodynamicProperty::Pressure, {fakeAllaireField});
        //(MPI_COMM_WORLD, "Finished initializing phase %lu\n", k);
    }
    //(MPI_COMM_WORLD, "Finished NStiffDecoder constructor\n");
}

static PetscInt cnt = 0;
#include <signal.h>
void ablate::finiteVolume::processes::NPhaseAllaireAdvection::NStiffDecoder::DecodeNPhaseAllaireState(DM subDM, const PetscReal *centroid, PetscInt dim, const PetscInt *uOff, const PetscReal *conservedValues,
                                                                                                                    const PetscReal *normal,       // The unit normal of the face
                                                                                                                    PetscReal *densityOut,         // Total density
                                                                                                                    PetscReal *densitykOut,        // Density of each phase
                                                                                                                    PetscReal *normalVelocityOut,  // Normal velocity
                                                                                                                    PetscReal *velocityOut,        // Velocity
                                                                                                                    PetscReal *internalEnergyOut,  // Total internal energy
                                                                                                                    PetscReal *internalEnergykOut, // Internal energy of each phase
                                                                                                                    PetscReal *aOut,               // Mixture speed of sound
                                                                                                                    PetscReal *akOut,              // Speed of sound of each phase
                                                                                                                    PetscReal *MkOut,              // Mach number of each phase
                                                                                                                    PetscReal *pOut,               // Total pressure
                                                                                                                    PetscReal *TkOut) {            // Phase temperature

//PetscBool debug = (dim==-2);
dim = PetscAbsInt(dim);

    std::size_t nPhases = eosk.size();

    // Declare all needed vectors and variables
    PetscReal *rhok, *Cpk, *gammak, *pik;
    DMGetWorkArray(subDM, nPhases, MPIU_REAL, &rhok)  >> utilities::PetscUtilities::checkError;
    DMGetWorkArray(subDM, nPhases, MPIU_REAL, &Cpk)  >> utilities::PetscUtilities::checkError;
    DMGetWorkArray(subDM, nPhases, MPIU_REAL, &gammak)  >> utilities::PetscUtilities::checkError;
    DMGetWorkArray(subDM, nPhases, MPIU_REAL, &pik)  >> utilities::PetscUtilities::checkError;

    const PetscReal *alphak = &conservedValues[uOff[ALPHAK_OFFSET]];
    const PetscReal *alphaRhok = &conservedValues[uOff[ALPHAKRHOK_OFFSET]];

    PetscReal rho = 0.0; // total density rho = sum_k (alpha_k*rho_k)
    for (std::size_t k = 0; k < nPhases; k++) {

        Cpk[k]    = eosk[k]->GetSpecificHeatCp();
        gammak[k] = eosk[k]->GetSpecificHeatRatio();
        pik[k]    = eosk[k]->GetReferencePressure();

        if (alphak[k] > 1e-15) rhok[k] = alphaRhok[k] / alphak[k];  // rho_k = (alpha_k*rho_k)/alpha_k
        else rhok[k] = 0;

        if (rhok[k] < 0) {
          printf("%lu\n", k);
          printf("%10s: %+e\n", "alphaRho", conservedValues[uOff[ALPHAKRHOK_OFFSET] + k]);
          printf("%10s: %+e\n", "alphak", alphak[k]);
          throw std::runtime_error("Negative density.\n");
        }
        rho += alphaRhok[k];
    }


    if (rho < PETSC_SMALL) { // This may be zero during initialization
      *densityOut = 0;
      *normalVelocityOut = 0;
      for (PetscInt d = 0; d < dim; d++) velocityOut[d] = 0;
      *internalEnergyOut = 0;
      *pOut = 0;
      for (std::size_t k = 0; k < nPhases; k++) {
          densitykOut[k] = 0;
          internalEnergykOut[k] = 0;
          akOut[k] = 0;
          MkOut[k] = 0;
          TkOut[k] = 0;
      }
      return;

    }

    const PetscReal *allaire = &conservedValues[uOff[ALLAIRE_OFFSET]];
    PetscReal rhoKE = 0.0;
    for (PetscInt d = 0; d < dim; d++) {
        rhoKE += PetscSqr(allaire[ablate::finiteVolume::NPhaseFlowFields::RHOU + d]);
    }
    PetscReal rhoIntE = allaire[ablate::finiteVolume::NPhaseFlowFields::RHOE] - 0.5*rhoKE/rho;

    PetscReal den = 0.0, num = 0;
    for (std::size_t k = 0; k < nPhases; k++) {
        num += alphak[k] * gammak[k] * pik[k] / (gammak[k] - 1.0);
        den += alphak[k] / (gammak[k] - 1.0);
    }

    // Final pressure calculation
//    const PetscReal p = static_cast<PetscReal>((rhoIntE - num) / den);
    const PetscReal p = ((rhoIntE - num) / den); // Shifted energy
//    const PetscReal p = ((rhoIntE) / den); // Shifted energy


    if (p < 0 || p > 1e12) {
      printf("%+e\n", rhoIntE);
      printf("%+e\t%+e\n", alphak[0], alphak[1]);
      printf("%+e\t%+e\n", rhok[0], rhok[1]);
      printf("%+e\n", rho);
      printf("%+f\t%+f\n", centroid[0], centroid[1]);
      printf("Negative pressure\n");
      printf("%+e\n", p);
      raise(SIGSEGV);
//      printf("%10s: %+e\n", "rhoIntE", rhoIntE);
//      printf("%10s: %+e\n", "num", num);
//      printf("%10s: %+e\n", "den", den);
      xexit("");

    }

     // Set output values
    *densityOut = rho;
    *normalVelocityOut = 0.0;
    for (PetscInt d = 0; d < dim; d++) {
        velocityOut[d] = allaire[ablate::finiteVolume::NPhaseFlowFields::RHOU + d]/rho;
        *normalVelocityOut += velocityOut[d] * normal[d];
    }
    *internalEnergyOut = rhoIntE/rho;
    *pOut = p;

    PetscReal a = 0; // Mixture speed of sound
    for (std::size_t k = 0; k < nPhases; k++) {
        if (rhok[k] > 0) {

            densitykOut[k] = rhok[k];

            // Compute internal energy per unit mass for phase k
            internalEnergykOut[k] = (p + gammak[k] * pik[k]) / ((gammak[k] - 1.0) * rhok[k]);

            rhoIntE += alphak[k] * rhok[k] * internalEnergykOut[k];

            // Compute temperature for phase k
            TkOut[k] = gammak[k] * (internalEnergykOut[k] - pik[k]/rhok[k]) / Cpk[k];

            // Compute speed of sound for phase k
            // Something weird happens when gammak[k] * (p + pik[k]) / rhok[k] evaluates to zero. The sqrt returns +inf
            akOut[k] = PetscSqrtReal(gammak[k] * (p + pik[k]) / rhok[k]);

             if (akOut[k] > 0) {
                MkOut[k] = (*normalVelocityOut) / akOut[k];

                /*
                    Mixture speed of sound from Pandare, Waltz, and Bakosi.
                    Note that on Pg. 884 they state "Note that Wood's speed of sound,47 which is more appropriate for
                    pressure-equilibrium multiphase flows, is not used in the examples shown in this work.
                    The pressure non-equilibrium speed of sound12,18 is used here.
                */
                a += alphak[k] * rhok[k] * akOut[k] * akOut[k];
//                a += alphak[k] / (rhok[k] * akOut[k] * akOut[k]);
            } else {
                MkOut[k] = 0.0;
            }


            if (internalEnergykOut[k] < 0) {
              printf("%lu\n", k);
              printf("%10s: %+e\n", "P", p);
              printf("%10s: %+e\n", "gammak", gammak[k]);
              printf("%10s: %+e\n", "pik", pik[k]);
              printf("%10s: %+e\n", "rhok", rhok[k]);
              printf("%+e\n", internalEnergykOut[k]);
              throw std::runtime_error("Negative energy.\n");
            }

            if (TkOut[k] < 0) {
              printf("%lu\n", k);
              printf("%10s: %+e\n", "gamma", gammak[k]);
              printf("%10s: %+e\n", "ek", internalEnergykOut[k]);
              printf("%10s: %+e\n", "pik", pik[k]);
              printf("%10s: %+e\n", "rhok", rhok[k]);
              printf("%10s: %+e\n", "cpk", Cpk[k]);
              printf("%+e\n", TkOut[k]);
              throw std::runtime_error("Negative temperature.\n");
            }



        } else {
            densitykOut[k] = 0.0;
            internalEnergykOut[k] = 0.0;
            TkOut[k] = 0.0;
            akOut[k] = 0.0;
            MkOut[k] = 0.0;
        }
    }
//    if (PetscAbsReal((*internalEnergyOut - rhoIntE/rho)/(*internalEnergyOut)) > 1e-4) {
//      printf("%+e\n%+e\n", *internalEnergyOut, rhoIntE/rho);
//      printf("%e\n", PetscAbsReal((*internalEnergyOut - rhoIntE/rho)/(*internalEnergyOut)));
//      throw std::runtime_error("Mismatch in total internal energy");
//    }

    if (a < PETSC_SMALL) {
      printf("%d\n", cnt);
      printf("%10s: %+e\n", "alpha0", alphak[0]);
      printf("%10s: %+e\n", "a0", akOut[0]);
      printf("%10s: %+e\n", "alpha1", alphak[1]);
      printf("%10s: %+e\n", "a1", akOut[1]);
      throw std::runtime_error("Speed of sound is too small!\n");
    }
//    *aOut = PetscSqrtReal(1/(rho*a)); // Mixture speed of sound
    *aOut = PetscSqrtReal(a / rho);

//    if (PetscIsNanReal(*aOut)) {
//      printf("%d\n", cnt);
//      printf("%+e\t%+e\n", alphak[0], alphak[1]);
//      printf("%+e\t%+e\n", rhok[0], rhok[1]);
//      printf("%+e\t%+e\n", akOut[0], akOut[1]);
//      printf("%+e\n", a);
//      printf("%+e\n", rho);
//      printf("%s::%d\n", __FILE__, __LINE__);
//      exit(0);
//    }

    DMRestoreWorkArray(subDM, nPhases, MPIU_REAL, &rhok)  >> utilities::PetscUtilities::checkError;
    DMRestoreWorkArray(subDM, nPhases, MPIU_REAL, &Cpk)  >> utilities::PetscUtilities::checkError;
    DMRestoreWorkArray(subDM, nPhases, MPIU_REAL, &gammak)  >> utilities::PetscUtilities::checkError;
    DMRestoreWorkArray(subDM, nPhases, MPIU_REAL, &pik)  >> utilities::PetscUtilities::checkError;
}

PetscErrorCode ablate::finiteVolume::processes::NPhaseAllaireAdvection::ZalesakTestSourceTerm(
    PetscInt dim, PetscReal time, const PetscFVCellGeom* cg, const PetscInt uOff[], const PetscScalar u[], const PetscInt aOff[], const PetscScalar a[], PetscScalar f[], void* ctx) {
    PetscFunctionBeginUser;

    auto* nPhase = static_cast<NPhaseAllaireAdvection*>(ctx);
    const std::size_t phases = nPhase->eosk.size();
    const PetscInt allaireOffset    = uOff[0];
    const PetscInt alphakrhokOffset = uOff[1];
    const PetscInt alphakOffset     = uOff[2];

    const PetscInt allaireOutOff    = 0;
    const PetscInt alphakrhokOutOff = 1 + dim;

    for (PetscInt i = 0; i < allaireOutOff + (1 + dim) + (PetscInt)phases - allaireOutOff; ++i) {
        f[i] = 0.0;
    }

    const PetscReal x = cg->centroid[0];
    const PetscReal y = (dim > 1) ? cg->centroid[1] : 0.0;
    const PetscReal u_target = 30.0 * (0.5 - y);
    const PetscReal v_target = 30.0 * (x - 0.5);
    const PetscReal v2_target = u_target * u_target + v_target * v_target;

    constexpr PetscReal rho_const = 1.0;
    constexpr PetscReal eps_const = 2.5;

    constexpr PetscReal dt = 1e-4;
    constexpr PetscReal inv_dt = 1.0 / dt;

    const PetscReal rhoU_current = u[allaireOffset + NPhaseFlowFields::RHOU];
    const PetscReal rhoV_current = (dim > 1) ? u[allaireOffset + NPhaseFlowFields::RHOV] : 0.0;
    const PetscReal rhoE_current = u[allaireOffset + NPhaseFlowFields::RHOE];

    const PetscReal rhoU_target = rho_const * u_target;
    const PetscReal rhoV_target = rho_const * v_target;
    const PetscReal rhoE_target = rho_const * (eps_const + 0.5 * v2_target);

    if (PetscIsInfOrNanReal(u_target) || PetscIsInfOrNanReal(v_target)) {
        PetscPrintf(MPI_COMM_WORLD, "ZALESAK TEST DEBUG: NaN/Inf velocity detected at time=%g, x=%g, y=%g\n", time, x, y);
    }

    f[allaireOutOff + NPhaseFlowFields::RHOU] = (rhoU_target - rhoU_current) * inv_dt;
    if (dim > 1) {
        f[allaireOutOff + NPhaseFlowFields::RHOV] = (rhoV_target - rhoV_current) * inv_dt;
    }
    f[allaireOutOff + NPhaseFlowFields::RHOE] = (rhoE_target - rhoE_current) * inv_dt;

    for (std::size_t k = 0; k < phases; ++k) {
        const PetscReal alphakrhok_target = u[alphakOffset + k] * rho_const;
        const PetscReal alphakrhok_current = u[alphakrhokOffset + k];
        f[alphakrhokOutOff + (PetscInt)k] = (alphakrhok_target - alphakrhok_current) * inv_dt;
    }
    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::NPhaseAllaireAdvection, "", ARG(ablate::eos::EOS, "eos", "must be nPhase"),
         OPT(ablate::parameters::Parameters, "parameters", "the parameters used by advection: cfl(.5)"),
         ARG(ablate::finiteVolume::fluxCalculator::FluxCalculator, "fluxCalculatorNStiff", ""));
