#include "nPhaseAllaireAdvectionOld.hpp"

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

#include "intSharp.hpp"

#include <signal.h>

#define NOTE0EXIT(S, ...) {PetscFPrintf(MPI_COMM_WORLD, stderr,                                     \
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

//static PetscBool oldDebugFlag = PETSC_FALSE;

//two phase precedent
//eosTwoPhase encapsulates thermo properties of the two phases (eos::TwoPhase --> eos::NPhase)
//parametersIn contains the cfl number
//std move transfers ownership of eosTwoPhase, fluxCalculatorXX to this class

//parameters checks parametersIn. If null then it returns an empty set


ablate::finiteVolume::processes::NPhaseAllaireAdvectionOld::NPhaseAllaireAdvectionOld(std::shared_ptr<eos::EOS> eosNPhase, const std::shared_ptr<parameters::Parameters> &parametersIn,
                                                                                std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorNStiff)
    : eosNPhase(std::move(eosNPhase)),
      fluxCalculatorNStiff(std::move(fluxCalculatorNStiff)) {
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
    //(MPI_COMM_WORLD, "phases = %lu\n", phases);
    eosk.resize(phases);

    for (std::size_t k=0; k<phases; k++) {
        auto phaseEOS = nPhaseEOS->GetEOSk(k);
        auto kthEOS = std::dynamic_pointer_cast<eos::KthStiffenedGas>(phaseEOS);
        if (!kthEOS) {
            throw std::invalid_argument("Each phase EOS must be of type KthStiffenedGas");
        }
        eosk[k] = kthEOS;
        //(MPI_COMM_WORLD, "eosk[%lu] initialized\n", k);
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
        PetscPrintf(MPI_COMM_WORLD, "[NPhaseAllaireAdvectionOld] Zalesak test enabled\n");
    }

    //(MPI_COMM_WORLD, "end of constructor\n");
}


ablate::finiteVolume::processes::NPhaseAllaireAdvectionOld::~NPhaseAllaireAdvectionOld() {
    // Destructor implementation
}

void ablate::finiteVolume::processes::NPhaseAllaireAdvectionOld::Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) {
    //(MPI_COMM_WORLD, "Starting Setup\n");
    //(MPI_COMM_WORLD, "eosk.size() in Setup = %lu\n", eosk.size());

    // Before each step, compute the alpha
//    auto multiphasePreStage = std::bind(&ablate::finiteVolume::processes::NPhaseAllaireAdvectionOld::MultiphaseFlowPreStage, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
//    flow.RegisterPreStage(multiphasePreStage);

    ablate::domain::SubDomain& subDomain = flow.GetSubDomain();
    //(MPI_COMM_WORLD, "Creating decoder with %lu phases\n", eosk.size());

    // Create the decoder based upon the eoses
    decoder = CreateNPhaseDecoder(subDomain.GetDimensions(), eosk);
    //(MPI_COMM_WORLD, "Decoder created\n");

    // Currently, no option for species advection
//    flow.RegisterRHSFunction(CompressibleFlowCompleteFlux, this);
    // flow.RegisterRHSFunction(NPhaseFlowCompleteFlux, this); //necessary?

    flow.RegisterRHSFunction(NPhaseFlowComputeAllaireFlux, this, {NPhaseFlowFields::ALLAIRE_FIELD}, {ALPHAK, ALPHAKRHOK, NPhaseFlowFields::ALLAIRE_FIELD}, {});
    //register the alphakrhok and alphak fluxes
    flow.RegisterRHSFunction(NPhaseFlowComputeAlphakFlux, this, {ALPHAK}, {ALPHAK, ALPHAKRHOK, NPhaseFlowFields::ALLAIRE_FIELD}, {});
    flow.RegisterRHSFunction(NPhaseFlowComputeAlphakRhokFlux, this, {ALPHAKRHOK}, {ALPHAK, ALPHAKRHOK, NPhaseFlowFields::ALLAIRE_FIELD}, {});


    // Register Zalesak test as source term if enabled
    if (zalesakTest) {
        PetscPrintf(MPI_COMM_WORLD, "[NPhaseAllaireAdvectionOld::Setup] About to register Zalesak test\n");
        PetscPrintf(MPI_COMM_WORLD, "[NPhaseAllaireAdvectionOld::Setup] ALPHAK field offset: %d\n", subDomain.GetField(ALPHAK).offset);
        PetscPrintf(MPI_COMM_WORLD, "[NPhaseAllaireAdvectionOld::Setup] ALPHAKRHOK field offset: %d\n", subDomain.GetField(ALPHAKRHOK).offset);
        PetscPrintf(MPI_COMM_WORLD, "[NPhaseAllaireAdvectionOld::Setup] ALLAIRE_FIELD offset: %d\n", subDomain.GetField(NPhaseFlowFields::ALLAIRE_FIELD).offset);

        flow.RegisterRHSFunction(static_cast<ablate::finiteVolume::CellInterpolant::PointFunction>(ZalesakTestSourceTerm),
            this,
            {NPhaseFlowFields::ALLAIRE_FIELD, ALPHAKRHOK},    // Outputs
            {NPhaseFlowFields::ALLAIRE_FIELD, ALPHAKRHOK, ALPHAK},  // Inputs
            {});
        PetscPrintf(MPI_COMM_WORLD, "[NPhaseAllaireAdvectionOld::Setup] Zalesak test registered successfully\n");
    }

    flow.RegisterComputeTimeStepFunction(ComputeCflTimeStep, &timeStepData, "cfl");
    timeStepData.computeSpeedOfSound = eosNPhase->GetThermodynamicFunction(eos::ThermodynamicProperty::SpeedOfSound, subDomain.GetFields());



    if (subDomain.ContainsField(NPhaseFlowFields::PRESSURE) && (subDomain.GetField(NPhaseFlowFields::PRESSURE).location == ablate::domain::FieldLocation::AUX)) {
      auxUpdateFields.push_back(NPhaseFlowFields::PRESSURE);
    }

    if (subDomain.ContainsField(NPhaseFlowFields::UI) && (subDomain.GetField(NPhaseFlowFields::UI).location == ablate::domain::FieldLocation::AUX)) {
      auxUpdateFields.push_back(NPhaseFlowFields::UI);
    }

    if (subDomain.ContainsField(NPhaseFlowFields::TK) && (subDomain.GetField(NPhaseFlowFields::TK).location == ablate::domain::FieldLocation::AUX)) {
      auxUpdateFields.push_back(NPhaseFlowFields::TK);
    }
    if (subDomain.ContainsField(NPhaseFlowFields::RHO) && (subDomain.GetField(NPhaseFlowFields::RHO).location == ablate::domain::FieldLocation::AUX)) {
      auxUpdateFields.push_back(NPhaseFlowFields::RHO);
    }
    if (subDomain.ContainsField(NPhaseFlowFields::RHOK) && (subDomain.GetField(NPhaseFlowFields::RHOK).location == ablate::domain::FieldLocation::AUX)) {
      auxUpdateFields.push_back(NPhaseFlowFields::RHOK);
    }
    if (subDomain.ContainsField(NPhaseFlowFields::EPSILON) && (subDomain.GetField(NPhaseFlowFields::EPSILON).location == ablate::domain::FieldLocation::AUX)) {
      auxUpdateFields.push_back(NPhaseFlowFields::EPSILON);
    }
    if (subDomain.ContainsField(NPhaseFlowFields::EPSILONK) && (subDomain.GetField(NPhaseFlowFields::EPSILONK).location == ablate::domain::FieldLocation::AUX)) {
      auxUpdateFields.push_back(NPhaseFlowFields::EPSILONK);
    }
    if (subDomain.ContainsField(NPhaseFlowFields::SOSK) && (subDomain.GetField(NPhaseFlowFields::SOSK).location == ablate::domain::FieldLocation::AUX)) {
      auxUpdateFields.push_back(NPhaseFlowFields::SOSK);
    }

    // include Aij interface indicator field if present
    if (subDomain.ContainsField(NPhaseFlowFields::AIJ) && (subDomain.GetField(NPhaseFlowFields::AIJ).location == ablate::domain::FieldLocation::AUX)) {
      auxUpdateFields.push_back(NPhaseFlowFields::AIJ);
    }

    // if (subDomain.ContainsField(ALPHAK) && (subDomain.GetField(ALPHAK).location == ablate::domain::FieldLocation::AUX)) {
    //   auxUpdateFields.push_back(ALPHAK);
    // }

    // if (subDomain.ContainsField(ALPHAKRHOK) && (subDomain.GetField(ALPHAKRHOK).location == ablate::domain::FieldLocation::AUX)) {
    //   auxUpdateFields.push_back(ALPHAKRHOK);
    // }

    if (auxUpdateFields.size() > 0) {
      flow.RegisterAuxFieldUpdate(
            UpdateAuxFieldsNPhase, this, auxUpdateFields, {ALPHAK, ALPHAKRHOK, NPhaseFlowFields::ALLAIRE_FIELD});
    }

    //initialize intsharp instance in setup;
    // we will then compute the intsharp term in the prestage and iteratively add it to the volume fraction
    // until a sufficient volume fraction gradient/sharpness is achieved
    // std::shared_ptr<ablate::finiteVolume::processes::IntSharp> intSharpProcess;
    // auto intSharpProcess = std::make_shared<ablate::finiteVolume::processes::IntSharp>(0, 0.001, false);
    // intSharpProcess->Initialize(flow);

    //(MPI_COMM_WORLD, "Setup complete\n");

}
#include <signal.h>
// Update the volume fraction, velocity, temperature, pressure fields, and gas density fields (if they exist).
PetscErrorCode ablate::finiteVolume::processes::NPhaseAllaireAdvectionOld::UpdateAuxFieldsNPhase(PetscReal time, PetscInt dim, const PetscFVCellGeom *cellGeom, const PetscInt uOff[],
                                                                                                   const PetscScalar *conservedValues, const PetscInt aOff[], PetscScalar *auxField, void *ctx) {
    PetscFunctionBeginUser;
    //(MPI_COMM_WORLD, "UpdateAuxFieldsNPhase: Starting function\n");

    if (!auxField) {
        //(MPI_COMM_WORLD, "UpdateAuxFieldsNPhase: auxField is null, returning\n");
        PetscFunctionReturn(0);
    }

    auto nPhaseAllaireAdvection = (NPhaseAllaireAdvectionOld *)ctx;
    //(MPI_COMM_WORLD, "UpdateAuxFieldsNPhase: Got context\n");

    // For cell center, the norm is unity
    PetscReal norm[3];
    norm[0] = 1;
    norm[1] = 1;
    norm[2] = 1;

    PetscReal density = 1.0;
    std::vector<PetscReal> densityk;
    PetscReal normalVelocity = 0.0;  // uniform velocity in cell
    PetscReal velocity[3] = {0.0, 0.0, 0.0};
    PetscReal internalEnergy = 0.0;
    std::vector<PetscReal> internalEnergyk;
    std::vector<PetscReal> ak;
    std::vector<PetscReal> Mk;
    PetscReal p = 0.0;  // pressure equilibrium
    std::vector<PetscReal> Tk;
    std::vector<PetscReal> alphak;

    if (conservedValues) {
        //(MPI_COMM_WORLD, "UpdateAuxFieldsNPhase: About to decode state\n");
        try {
            nPhaseAllaireAdvection->decoder->DecodeNPhaseAllaireState(
                dim, uOff, conservedValues, norm, &density, &densityk, &normalVelocity, velocity, &internalEnergy, &internalEnergyk, &ak,
                &Mk, &p, &Tk, &alphak);
            //(MPI_COMM_WORLD, "UpdateAuxFieldsNPhase: Successfully decoded state\n");
        } catch (const std::exception& e) {
            //(MPI_COMM_WORLD, "UpdateAuxFieldsNPhase: Error in DecodeNPhaseAllaireState: %s\n", e.what());
            throw;
        }

        for (PetscInt d = 0; d < dim; d++) {
            velocity[d] = conservedValues[NPhaseFlowFields::RHOU + d] / density;
        }
        //(MPI_COMM_WORLD, "UpdateAuxFieldsNPhase: Computed velocities\n");
    }

    auto fields = nPhaseAllaireAdvection->auxUpdateFields.data();
    //(MPI_COMM_WORLD, "UpdateAuxFieldsNPhase: Starting field updates, number of fields: %zu\n", nPhaseAllaireAdvection->auxUpdateFields.size());

    for (std::size_t f = 0; f < nPhaseAllaireAdvection->auxUpdateFields.size(); ++f) {
        // (MPI_COMM_WORLD, "UpdateAuxFieldsNPhase: Processing field %s\n", fields[f].c_str());


        if (fields[f] == NPhaseFlowFields::UI) {
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
            std::size_t nPhases = nPhaseAllaireAdvection->eosk.size();
            PetscInt idx = 0;
            for (std::size_t i = 0; i < nPhases; i++) {
                for (std::size_t j = i + 1; j < nPhases; j++) {
                    PetscReal denom = alphak[i] + alphak[j];
                    PetscReal value = (denom > PETSC_SMALL) ? (alphak[i] / denom) : 1.0;
                    auxField[aOff[f] + idx] = value;
                    idx++;
                }
            }
        }
        // else if (fields[f] == NPhaseFlowFields::ALPHAKRHOK) {
        //     for (std::size_t k = 0; k < nPhaseAllaireAdvection->eosk.size(); k++) {
        //         auxField[aOff[f] + k] = densityk[k];
        //     }
        // }
        // else if (fields[f] == NPhaseFlowFields::ALPHAK) {
        //     for (std::size_t k = 0; k < nPhaseAllaireAdvection->eosk.size(); k++) {
        //         auxField[aOff[f] + k] = ak[k];
        //     }
        // }
        //(MPI_COMM_WORLD, "UpdateAuxFieldsNPhase: Completed field %s\n", fields[f].c_str());
    }

    //(MPI_COMM_WORLD, "UpdateAuxFieldsNPhase: Completed all field updates\n");
    PetscFunctionReturn(0);
}
#include <iostream>


PetscErrorCode ablate::finiteVolume::processes::NPhaseAllaireAdvectionOld::MultiphaseFlowPreStage(TS flowTs, ablate::solver::Solver &solver, PetscReal stagetime) {
    PetscFunctionBegin;

    printf("%s::%d\n", __FILE__, __LINE__); exit(0);
#if 0
    // Add debug print at start of prestage
    // (MPI_COMM_WORLD, "MultiphaseFlowPreStage - Starting pre-stage update at time %g\n", stagetime);

    // Get flow field data
    //(MPI_COMM_WORLD, "2. About to dynamic_cast solver\n");
    const auto &fvSolver = dynamic_cast<ablate::finiteVolume::FiniteVolumeSolver &>(solver);
    //(MPI_COMM_WORLD, "3. Cast successful\n");

    ablate::domain::Range cellRange;
    //(MPI_COMM_WORLD, "4. About to GetCellRangeWithoutGhost\n");
    fvSolver.GetCellRangeWithoutGhost(cellRange);
    //(MPI_COMM_WORLD, "5. Got cell range\n");

    PetscInt dim;
    //(MPI_COMM_WORLD, "6. About to get dimension\n");
    PetscCall(DMGetDimension(fvSolver.GetSubDomain().GetDM(), &dim));
    //(MPI_COMM_WORLD, "7. Got dimension\n");

    //(MPI_COMM_WORLD, "8. About to get field offsets\n");
    const auto &allaireOffset = fvSolver.GetSubDomain().GetField(NPhaseFlowFields::ALLAIRE_FIELD).offset;
    const auto &alphakOffset = fvSolver.GetSubDomain().GetField(ALPHAK).offset;
    const auto &alphakRhokOffset = fvSolver.GetSubDomain().GetField(ALPHAKRHOK).offset;
    // (MPI_COMM_WORLD, "alphakOffset: %d, alphakRhokOffset: %d, allaireOffset: %d\n", alphakOffset, alphakRhokOffset, allaireOffset);
    //(MPI_COMM_WORLD, "9. Got field offsets\n");

    //(MPI_COMM_WORLD, "10. About to get DM\n");
    DM dm = fvSolver.GetSubDomain().GetDM();
    //(MPI_COMM_WORLD, "11. Got DM\n");

    //(MPI_COMM_WORLD, "12. About to get solution\n");
    Vec globFlowVec;
    PetscCall(TSGetSolution(flowTs, &globFlowVec));
    //(MPI_COMM_WORLD, "13. Got solution\n");

    //(MPI_COMM_WORLD, "14. About to get array\n");
    PetscScalar *flowArray;
    PetscCall(VecGetArray(globFlowVec, &flowArray));
    //(MPI_COMM_WORLD, "15. Got array\n");

    PetscInt uOff[3];
    uOff[0] = alphakOffset;
    uOff[1] = alphakRhokOffset;
    uOff[2] = allaireOffset;

    //get the rhs vector
    Vec locFVec;
    PetscCall(DMGetLocalVector(dm, &locFVec));
    PetscCall(VecZeroEntries(locFVec));

    // For cell center, the norm is unity
    PetscReal norm[3];
    norm[0] = 1;
    norm[1] = 1;
    norm[2] = 1;

    for (PetscInt i = cellRange.start; i < cellRange.end; ++i) {
        const PetscInt cell = cellRange.GetPoint(i);
        PetscScalar *allFields = nullptr;
        DMPlexPointLocalRef(dm, cell, flowArray, &allFields) >> utilities::PetscUtilities::checkError;

        // Add debug prints for cell values
        // if (allFields[allaireOffset + NPhaseFlowFields::RHOU] > PETSC_SMALL || allFields[allaireOffset + NPhaseFlowFields::RHOV] > PETSC_SMALL) {
        //     (MPI_COMM_WORLD, "Cell %d - Current state:\n", cell);
        //     (MPI_COMM_WORLD, "  rhou=%g, rhov=%g\n",
        //                allFields[allaireOffset + NPhaseFlowFields::RHOU],
        //                allFields[allaireOffset + NPhaseFlowFields::RHOV]);
        // }

        // allFields[alphakOffset] +=

        // auto density = allFields[ablate::finiteVolume::CompressibleFlowFields::RHO];
        auto density = 0.0;
        //density is sumk alphak * rhok
        for (std::size_t k = 0; k < eosk.size(); k++) {
            density += allFields[alphakRhokOffset + k];
        }
        // (MPI_COMM_WORLD, "density: %f\n", density);
        PetscReal velocity[3];
        for (PetscInt d = 0; d < dim; d++) {
            velocity[d] = allFields[ablate::finiteVolume::NPhaseFlowFields::RHOU + d] / density;
        }

        // // Decode state
        // std::vector<PetscReal> densityk;
        // PetscReal normalVelocity;
        // PetscReal internalEnergy;
        // std::vector<PetscReal> internalEnergyk;
        // std::vector<PetscReal> ak;
        // std::vector<PetscReal> Mk;
        // PetscReal p;
        // std::vector<PetscReal> tk;
        // std::vector<PetscReal> alphak;

        // decoder->DecodeNPhaseAllaireState(
        //     dim, uOff, allFields, norm, &density, &densityk, &normalVelocity, velocity, &internalEnergy, &internalEnergyk, &ak, &Mk, &p, &tk, &alphak);

        // // update all alphak
        // for (std::size_t k = 0; k < eosk.size(); k++) {
        //     allFields[uOff[0] + k] = alphak[k];
        // }

    }

    //restore
    PetscCall(DMRestoreLocalVector(dm, &locFVec));
    PetscCall(VecRestoreArray(globFlowVec, &flowArray));

    // clean up
    fvSolver.RestoreRange(cellRange);

    // Add debug print at end of prestage
    // (MPI_COMM_WORLD, "MultiphaseFlowPreStage - Completed pre-stage update\n");
#endif
    PetscFunctionReturn(0);
}
double ablate::finiteVolume::processes::NPhaseAllaireAdvectionOld::ComputeCflTimeStep(TS ts, ablate::finiteVolume::FiniteVolumeSolver &flow, void *ctx) {
    // Get the dm and current solution vector

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
    auto allaireID = flow.GetSubDomain().GetField(ablate::finiteVolume::NPhaseFlowFields::ALLAIRE_FIELD).id;

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
PetscErrorCode ablate::finiteVolume::processes::NPhaseAllaireAdvectionOld::NPhaseFlowCompleteFlux(const ablate::finiteVolume::FiniteVolumeSolver &flow, DM dm, PetscReal time, Vec locXVec, Vec locFVec, void* ctx) {

  PetscFunctionBeginUser;
//  auto flow = (ablate::finiteVolume::FiniteVolumeSolver *)ctx;
//  auto twoPhaseEulerAdvection = (TwoPhaseEulerAdvection *)ctx;
  NOTE0EXIT("");

  PetscFunctionReturn(0);

}

//FILE *f2 = fopen("fluxOld.txt", "w");
//static PetscInt oldCnt = 0;

PetscErrorCode ablate::finiteVolume::processes::NPhaseAllaireAdvectionOld::NPhaseFlowComputeAllaireFlux(PetscInt dim, const PetscFVFaceGeom *fg, const PetscInt *uOff, const PetscScalar *fieldL,
                                                                                                         const PetscScalar *fieldR, const PetscInt *aOff, const PetscScalar *auxL,
                                                                                                         const PetscScalar *auxR, PetscScalar *flux, void *ctx) {
    PetscFunctionBeginUser;
//++oldCnt;

    auto nPhaseAllaireAdvection = (NPhaseAllaireAdvectionOld *)ctx;

    // Compute the norm of cell face
    PetscReal norm[3];
    NormVector(dim, fg->normal, norm);
    const PetscReal areaMag = MagVector(dim, fg->normal);

    // Decode left and right states
    PetscReal densityL = 0.0;
    std::vector<PetscReal> densityk_L;
    PetscReal normalVelocityL = 0.0;
    PetscReal velocityL[3] = {0.0, 0.0, 0.0};
    PetscReal internalEnergyL = 0.0;
    std::vector<PetscReal> internalEnergyk_L;
    std::vector<PetscReal> ak_L;
    std::vector<PetscReal> Mk_L;
    PetscReal pL = 0.0;
    std::vector<PetscReal> tk_L;
    std::vector<PetscReal> alphak_L;
    std::vector<PetscReal> alphakRhok_L;
    nPhaseAllaireAdvection->decoder->DecodeNPhaseAllaireState(dim, uOff, fieldL, norm,
                                                              &densityL, &densityk_L, &normalVelocityL, velocityL,
                                                              &internalEnergyL, &internalEnergyk_L, &ak_L, &Mk_L,
                                                              &pL, &tk_L, &alphak_L);

    PetscReal densityR = 0.0;
    std::vector<PetscReal> densityk_R;
    PetscReal normalVelocityR = 0.0;
    PetscReal velocityR[3] = {0.0, 0.0, 0.0};
    PetscReal internalEnergyR = 0.0;
    std::vector<PetscReal> internalEnergyk_R;
    std::vector<PetscReal> ak_R;
    std::vector<PetscReal> Mk_R;
    PetscReal pR = 0.0;
    std::vector<PetscReal> tk_R;
    std::vector<PetscReal> alphak_R;
    std::vector<PetscReal> alphakRhok_R;
    nPhaseAllaireAdvection->decoder->DecodeNPhaseAllaireState(dim, uOff, fieldR, norm,
                                                              &densityR, &densityk_R, &normalVelocityR, velocityR,
                                                              &internalEnergyR, &internalEnergyk_R, &ak_R, &Mk_R,
                                                              &pR, &tk_R, &alphak_R);

    // Compute effective speeds of sound
    PetscReal aL = 0.0;
    for (std::size_t k = 0; k < nPhaseAllaireAdvection->eosk.size(); k++) {
        if (ak_L[k] > 0.0) {
            aL += alphak_L[k] / ak_L[k];
        }
    }
    aL = 1.0 / aL;

    PetscReal aR = 0.0;
    for (std::size_t k = 0; k < nPhaseAllaireAdvection->eosk.size(); k++) {
        if (ak_R[k] > 0.0) {
            aR += alphak_R[k] / ak_R[k];
        }
    }
    aR = 1.0 / aR;


    //initialize alphakrhok_L and alphakrhok_R as having size of eosk
    alphakRhok_L.resize(nPhaseAllaireAdvection->eosk.size());
    alphakRhok_R.resize(nPhaseAllaireAdvection->eosk.size());
    // Compute alphakRhok
    for (std::size_t k = 0; k < nPhaseAllaireAdvection->eosk.size(); k++) {
        alphakRhok_L[k] = alphak_L[k] * densityk_L[k];
        alphakRhok_R[k] = alphak_R[k] * densityk_R[k];
    }


    // Create and compute full flux vector
    fluxCalculator::FullFluxVector fluxVec(nPhaseAllaireAdvection->eosk.size());
    if (nPhaseAllaireAdvection->fluxCalculatorNStiff->ComputeFullFluxVector(
            nPhaseAllaireAdvection->fluxCalculatorNStiff->GetFluxCalculatorContext(),
            normalVelocityL, aL, densityL, pL,
            normalVelocityR, aR, densityR, pR,
            dim, fg->normal, areaMag,
            velocityL, velocityR,
            internalEnergyL, internalEnergyR,
            nPhaseAllaireAdvection->eosk.size(),
            alphak_L.data(), alphak_R.data(),
            alphakRhok_L.data(), alphakRhok_R.data(),
            &fluxVec)) {

//fprintf(f2, "%+e\t%+e\t%+e\t", fluxVec.energyFlux, fluxVec.momentumFlux[0], fluxVec.momentumFlux[1]);

//if (oldCnt == 43511) {
////  printf("%+e\t%+e\t%+e\n", fluxVec.energyFlux, fluxVec.momentumFlux[0], fluxVec.momentumFlux[1]);
//  oldDebugFlag = PETSC_TRUE;
//  nPhaseAllaireAdvection->decoder->DecodeNPhaseAllaireState(dim, uOff, fieldR, norm,
//                                                              &densityR, &densityk_R, &normalVelocityR, velocityR,
//                                                              &internalEnergyR, &internalEnergyk_R, &ak_R, &Mk_R,
//                                                              &pR, &tk_R, &alphak_R);


////  printf("%+e\t%+e\t%+e\t%+e\t%+e\t%+e\t%+e\n", fieldR[0], fieldR[1], fieldR[2], fieldR[3], fieldR[4], fieldR[5], fieldR[6]);
////  printf("%+e\t%+e\t%+e\t%+e\t", normalVelocityL, aL, densityL, pL);
////  printf("%+e\t%+e\t%+e\t%+e\n", normalVelocityR, aR, densityR, pR);
//  printf("%s::%d\n", __FILE__, __LINE__);
//  exit(0);
//}

        // Use the computed flux vector directly
        flux[NPhaseFlowFields::RHOE] = fluxVec.energyFlux;
        for (PetscInt d = 0; d < dim; d++) {
            flux[NPhaseFlowFields::RHOU + d] = fluxVec.momentumFlux[d];
        }
        // Compute alphakRhok, alphak


    } else {
        // Fall back to original interface if full flux vector not supported
        PetscPrintf(MPI_COMM_WORLD, "WARNING, falling back to original interface\n");
        PetscReal massFlux = 0.0, p12 = 0.0;
        fluxCalculator::Direction direction = nPhaseAllaireAdvection->fluxCalculatorNStiff->GetFluxCalculatorFunction()(
            nPhaseAllaireAdvection->fluxCalculatorNStiff->GetFluxCalculatorContext(),
            normalVelocityL, aL, densityL, pL,
            normalVelocityR, aR, densityR, pR,
            &massFlux, &p12);

        // Use direction to determine which state to use
        PetscReal vel[3] = {0.0, 0.0, 0.0};
        PetscReal internalEnergy = 0.0, density = 0.0;

        if (direction == fluxCalculator::LEFT) {
            internalEnergy = internalEnergyL;
            density = densityL;
            PetscArraycpy(vel, velocityL, dim);
        } else if (direction == fluxCalculator::RIGHT) {
            internalEnergy = internalEnergyR;
            density = densityR;
            PetscArraycpy(vel, velocityR, dim);
        } else {
            internalEnergy = 0.5*(internalEnergyL + internalEnergyR);
            density = 0.5*(densityL + densityR);
            for (PetscInt d = 0; d < dim; d++) vel[d] = 0.5*(velocityL[d] + velocityR[d]);
        }

        PetscReal velMag = MagVector(dim, vel);
        PetscReal H = internalEnergy + 0.5 * velMag * velMag + p12 / density;

        flux[NPhaseFlowFields::RHOE] = H * massFlux * areaMag;
        for (PetscInt d = 0; d < dim; d++) {
            flux[NPhaseFlowFields::RHOU + d] = vel[d] * massFlux * areaMag + p12 * fg->normal[d] * areaMag;
        }
    }

    PetscFunctionReturn(0);
}

//create an analog to NPhaseFlowComputeAlphakRhokFlux
PetscErrorCode ablate::finiteVolume::processes::NPhaseAllaireAdvectionOld::NPhaseFlowComputeAlphakFlux(PetscInt dim, const PetscFVFaceGeom *fg, const PetscInt uOff[], const PetscScalar fieldL[], const PetscScalar fieldR[], const PetscInt aOff[],
                                                        const PetscScalar auxL[], const PetscScalar auxR[], PetscScalar *flux, void *ctx) {
    PetscFunctionBeginUser;

    auto nPhaseAllaireAdvection = (NPhaseAllaireAdvectionOld *)ctx;

    // Compute the norm of cell face
    PetscReal norm[3];
    NormVector(dim, fg->normal, norm);
    const PetscReal areaMag = MagVector(dim, fg->normal);

    // Decode left and right states
    PetscReal densityL = 0.0;
    std::vector<PetscReal> densityk_L;
    PetscReal normalVelocityL = 0.0;
    PetscReal velocityL[3] = {0.0, 0.0, 0.0};
    PetscReal internalEnergyL = 0.0;
    std::vector<PetscReal> internalEnergyk_L;
    std::vector<PetscReal> ak_L;
    std::vector<PetscReal> Mk_L;
    PetscReal pL = 0.0;
    std::vector<PetscReal> tk_L;
    std::vector<PetscReal> alphak_L;
    std::vector<PetscReal> alphakRhok_L;
    nPhaseAllaireAdvection->decoder->DecodeNPhaseAllaireState(dim, uOff, fieldL, norm,
                                                              &densityL, &densityk_L, &normalVelocityL, velocityL,
                                                              &internalEnergyL, &internalEnergyk_L, &ak_L, &Mk_L,
                                                              &pL, &tk_L, &alphak_L);

    PetscReal densityR = 0.0;
    std::vector<PetscReal> densityk_R;
    PetscReal normalVelocityR = 0.0;
    PetscReal velocityR[3] = {0.0, 0.0, 0.0};
    PetscReal internalEnergyR = 0.0;
    std::vector<PetscReal> internalEnergyk_R;
    std::vector<PetscReal> ak_R;
    std::vector<PetscReal> Mk_R;
    PetscReal pR = 0.0;
    std::vector<PetscReal> tk_R;
    std::vector<PetscReal> alphak_R;
    std::vector<PetscReal> alphakRhok_R;
    nPhaseAllaireAdvection->decoder->DecodeNPhaseAllaireState(dim, uOff, fieldR, norm,
                                                              &densityR, &densityk_R, &normalVelocityR, velocityR,
                                                              &internalEnergyR, &internalEnergyk_R, &ak_R, &Mk_R,
                                                              &pR, &tk_R, &alphak_R);

    // Compute effective speeds of sound
    PetscReal aL = 0.0;
    for (std::size_t k = 0; k < nPhaseAllaireAdvection->eosk.size(); k++) {
        if (ak_L[k] > 0.0) {
            aL += alphak_L[k] / ak_L[k];
        }
    }
    aL = 1.0 / aL;

    PetscReal aR = 0.0;
    for (std::size_t k = 0; k < nPhaseAllaireAdvection->eosk.size(); k++) {
        if (ak_R[k] > 0.0) {
            aR += alphak_R[k] / ak_R[k];
        }
    }
    aR = 1.0 / aR;


    //initialize alphakrhok_L and alphakrhok_R as having size of eosk
    alphakRhok_L.resize(nPhaseAllaireAdvection->eosk.size());
    alphakRhok_R.resize(nPhaseAllaireAdvection->eosk.size());
    // Compute alphakRhok
    for (std::size_t k = 0; k < nPhaseAllaireAdvection->eosk.size(); k++) {
        alphakRhok_L[k] = alphak_L[k] * densityk_L[k];
        alphakRhok_R[k] = alphak_R[k] * densityk_R[k];
    }


    // Create and compute full flux vector
    fluxCalculator::FullFluxVector fluxVec(nPhaseAllaireAdvection->eosk.size());
    if (nPhaseAllaireAdvection->fluxCalculatorNStiff->ComputeFullFluxVector(
            nPhaseAllaireAdvection->fluxCalculatorNStiff->GetFluxCalculatorContext(),
            normalVelocityL, aL, densityL, pL,
            normalVelocityR, aR, densityR, pR,
            dim, fg->normal, areaMag,
            velocityL, velocityR,
            internalEnergyL, internalEnergyR,
            nPhaseAllaireAdvection->eosk.size(),
            alphak_L.data(), alphak_R.data(),
            alphakRhok_L.data(), alphakRhok_R.data(),
            &fluxVec)) {

//fprintf(f2, "%+e\t%+e\t", fluxVec.alphakFlux[0], fluxVec.alphakFlux[1]);

        // Use the computed flux vector directly
        for (std::size_t k = 0; k < nPhaseAllaireAdvection->eosk.size(); k++) {
            flux[k] = fluxVec.alphakFlux[k];
        }


        // PetscReal ustar[dim];
        // for (PetscInt d = 0; d < dim; d++) {
        //     ustar[d] = fluxVec.ustar[d];
        // }


    }
    PetscFunctionReturn(0);
}

//create an analog for nphaseflowcomputealphakrhokflux
PetscErrorCode ablate::finiteVolume::processes::NPhaseAllaireAdvectionOld::NPhaseFlowComputeAlphakRhokFlux(PetscInt dim, const PetscFVFaceGeom *fg, const PetscInt uOff[], const PetscScalar fieldL[], const PetscScalar fieldR[], const PetscInt aOff[],
                                                        const PetscScalar auxL[], const PetscScalar auxR[], PetscScalar *flux, void *ctx) {
        PetscFunctionBeginUser;

    auto nPhaseAllaireAdvection = (NPhaseAllaireAdvectionOld *)ctx;

    // Compute the norm of cell face
    PetscReal norm[3];
    NormVector(dim, fg->normal, norm);
    const PetscReal areaMag = MagVector(dim, fg->normal);

    // Decode left and right states
    PetscReal densityL = 0.0;
    std::vector<PetscReal> densityk_L;
    PetscReal normalVelocityL = 0.0;
    PetscReal velocityL[3] = {0.0, 0.0, 0.0};
    PetscReal internalEnergyL = 0.0;
    std::vector<PetscReal> internalEnergyk_L;
    std::vector<PetscReal> ak_L;
    std::vector<PetscReal> Mk_L;
    PetscReal pL = 0.0;
    std::vector<PetscReal> tk_L;
    std::vector<PetscReal> alphak_L;
    std::vector<PetscReal> alphakRhok_L;
    nPhaseAllaireAdvection->decoder->DecodeNPhaseAllaireState(dim, uOff, fieldL, norm,
                                                              &densityL, &densityk_L, &normalVelocityL, velocityL,
                                                              &internalEnergyL, &internalEnergyk_L, &ak_L, &Mk_L,
                                                              &pL, &tk_L, &alphak_L);

    PetscReal densityR = 0.0;
    std::vector<PetscReal> densityk_R;
    PetscReal normalVelocityR = 0.0;
    PetscReal velocityR[3] = {0.0, 0.0, 0.0};
    PetscReal internalEnergyR = 0.0;
    std::vector<PetscReal> internalEnergyk_R;
    std::vector<PetscReal> ak_R;
    std::vector<PetscReal> Mk_R;
    PetscReal pR = 0.0;
    std::vector<PetscReal> tk_R;
    std::vector<PetscReal> alphak_R;
    std::vector<PetscReal> alphakRhok_R;
    nPhaseAllaireAdvection->decoder->DecodeNPhaseAllaireState(dim, uOff, fieldR, norm,
                                                              &densityR, &densityk_R, &normalVelocityR, velocityR,
                                                              &internalEnergyR, &internalEnergyk_R, &ak_R, &Mk_R,
                                                              &pR, &tk_R, &alphak_R);

    // Compute effective speeds of sound
    PetscReal aL = 0.0;
    for (std::size_t k = 0; k < nPhaseAllaireAdvection->eosk.size(); k++) {
        if (ak_L[k] > 0.0) {
            aL += alphak_L[k] / ak_L[k];
        }
    }
    aL = 1.0 / aL;

    PetscReal aR = 0.0;
    for (std::size_t k = 0; k < nPhaseAllaireAdvection->eosk.size(); k++) {
        if (ak_R[k] > 0.0) {
            aR += alphak_R[k] / ak_R[k];
        }
    }
    aR = 1.0 / aR;


    //initialize alphakrhok_L and alphakrhok_R as having size of eosk
    alphakRhok_L.resize(nPhaseAllaireAdvection->eosk.size());
    alphakRhok_R.resize(nPhaseAllaireAdvection->eosk.size());
    // Compute alphakRhok
    for (std::size_t k = 0; k < nPhaseAllaireAdvection->eosk.size(); k++) {
        alphakRhok_L[k] = alphak_L[k] * densityk_L[k];
        alphakRhok_R[k] = alphak_R[k] * densityk_R[k];
    }


    // Create and compute full flux vector
    fluxCalculator::FullFluxVector fluxVec(nPhaseAllaireAdvection->eosk.size());
    if (nPhaseAllaireAdvection->fluxCalculatorNStiff->ComputeFullFluxVector(
            nPhaseAllaireAdvection->fluxCalculatorNStiff->GetFluxCalculatorContext(),
            normalVelocityL, aL, densityL, pL,
            normalVelocityR, aR, densityR, pR,
            dim, fg->normal, areaMag,
            velocityL, velocityR,
            internalEnergyL, internalEnergyR,
            nPhaseAllaireAdvection->eosk.size(),
            alphak_L.data(), alphak_R.data(),
            alphakRhok_L.data(), alphakRhok_R.data(),
            &fluxVec)) {

//fprintf(f2, "%+e\t%+e\n", fluxVec.alphakRhokFlux[0], fluxVec.alphakRhokFlux[1]);

//printf("%s::%d\n", __FILE__, __LINE__);exit(0);
        // Use the computed flux vector directly
        for (std::size_t k = 0; k < nPhaseAllaireAdvection->eosk.size(); k++) {
            flux[k] = fluxVec.alphakRhokFlux[k];
        }
    }
    PetscFunctionReturn(0);
}


std::shared_ptr<ablate::finiteVolume::processes::NPhaseAllaireAdvectionOld::NPhaseDecoder> ablate::finiteVolume::processes::NPhaseAllaireAdvectionOld::CreateNPhaseDecoder(
    PetscInt dim, const std::vector<std::shared_ptr<eos::EOS>> &eosk) {

    //(MPI_COMM_WORLD, "Entering CreateNPhaseDecoder\n");
    //(MPI_COMM_WORLD, "Input eosk size: %lu\n", eosk.size());

    // return std::make_shared<NStiffDecoder>(dim, eosk);
    std::vector<std::shared_ptr<ablate::eos::KthStiffenedGas>> stiffGases;
    //(MPI_COMM_WORLD, "Created kthStiffenedGas vector\n");

    for (const auto& eos : eosk) {
      //(MPI_COMM_WORLD, "Attempting to cast EOS to kthStiffenedGas\n");
      auto stiffGas = std::dynamic_pointer_cast<ablate::eos::KthStiffenedGas>(eos);
      if (!stiffGas) {
        //(MPI_COMM_WORLD, "Failed to cast EOS to kthStiffenedGas\n");
        throw std::invalid_argument("All EOSs must be kthStiffenedGas for NPhaseAllaireAdvectionOld");
      }
      //(MPI_COMM_WORLD, "Successfully cast EOS to kthStiffenedGas\n");
      stiffGases.push_back(stiffGas);
    }
    //(MPI_COMM_WORLD, "Created stiffGases vector, size: %lu\n", stiffGases.size());
    auto decoder = std::make_shared<NStiffDecoder>(dim, stiffGases); //this is where the error is
    //(MPI_COMM_WORLD, "Successfully created NStiffDecoder\n");
    return decoder;
}


#include <signal.h>

ablate::finiteVolume::processes::NPhaseAllaireAdvectionOld::NStiffDecoder::NStiffDecoder(PetscInt dim, const std::vector<std::shared_ptr<eos::KthStiffenedGas>> &eosk)
    : eosk(eosk) {
    //(MPI_COMM_WORLD, "Starting NStiffDecoder constructor\n");

    std::size_t phases = eosk.size();
    //(MPI_COMM_WORLD, "Input eosk size: %lu\n", phases);

    // Create the fake euler field
    //(MPI_COMM_WORLD, "Creating fake Allaire field\n");
    auto fakeAllaireField = ablate::domain::Field{.name = NPhaseFlowFields::ALLAIRE_FIELD,
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

void ablate::finiteVolume::processes::NPhaseAllaireAdvectionOld::NStiffDecoder::DecodeNPhaseAllaireState(PetscInt dim, const PetscInt *uOff, const PetscReal *conservedValues,
                                                                                                                        const PetscReal *normal, PetscReal *densityOut, std::vector<PetscReal> *densitykOut, PetscReal *normalVelocityOut, PetscReal *velocityOut,
                                                                                                                        PetscReal *internalEnergyOut, std::vector<PetscReal> *internalEnergykOut, std::vector<PetscReal> *akOut, std::vector<PetscReal> *MkOut,
                                                                                                                         PetscReal *pOut, std::vector<PetscReal> *TkOut, std::vector<PetscReal> *alphakOut) {
    // Add debug prints for input conserved values
    //only print if the conserved values are not zero
    // if (conservedValues[uOff[2] + NPhaseFlowFields::RHOU] > PETSC_SMALL || conservedValues[uOff[2] + NPhaseFlowFields::RHOV] > PETSC_SMALL) {
    //   (MPI_COMM_WORLD, "DecodeNPhaseAllaireState - Input conserved values:\n");
    //   (MPI_COMM_WORLD, "  rhou=%g, rhov=%g\n",
    //             conservedValues[uOff[2] + NPhaseFlowFields::RHOU],
    //             conservedValues[uOff[2] + NPhaseFlowFields::RHOV]);
    // }

    std::size_t phases = eosk.size();
    densitykOut->resize(phases);
    internalEnergykOut->resize(phases);
    akOut->resize(phases);
    MkOut->resize(phases);
    TkOut->resize(phases);
    alphakOut->resize(phases);


    const int ALPHAK_FIELD = 0;
    const int ALPHAKRHOK_FIELD = 1; //these are correct
    const int ALLAIRE_FIELD = 2;


    const PetscInt ALPHAK_OFFSET = uOff[ALPHAK_FIELD];
    const PetscInt ALPHAKRHOK_OFFSET = uOff[ALPHAKRHOK_FIELD];
    const PetscInt ALLAIRE_OFFSET = uOff[ALLAIRE_FIELD];

    //(MPI_COMM_WORLD, "field offsets: allaire=%d, alphak=%d, alphakrhok=%d\n", ALLAIRE_OFFSET, ALPHAK_OFFSET, ALPHAKRHOK_OFFSET);

    //check to make sure the offsets are correct
    //loop over phases
    for (std::size_t k = 0; k < phases; k++) {
        //(MPI_COMM_WORLD, "conservedValues[ALPHAKRHOK_OFFSET + %lu]: %f\n", k, conservedValues[ALPHAKRHOK_OFFSET + k]);
        //(MPI_COMM_WORLD, "conservedValues[ALPHAK_OFFSET + %lu]: %f\n", k, conservedValues[ALPHAK_OFFSET + k]);
    }
    for (PetscInt i = 0; i < dim + 1; i++) {
        //(MPI_COMM_WORLD, "conservedValues[ALLAIRE_OFFSET + %d]: %f\n", i, conservedValues[ALLAIRE_OFFSET + i]);
    }

    // Declare all needed vectors and variables
    std::vector<PetscReal> rhok(phases);
    std::vector<PetscReal> ek(phases);
    std::vector<PetscReal> Tk(phases);
    std::vector<PetscReal> alphak(phases);
    std::vector<PetscReal> Cpk(phases);
    std::vector<PetscReal> gammak(phases);
    std::vector<PetscReal> pik(phases);
    std::vector<PetscReal> ck(phases);
    std::vector<PetscReal> Mk(phases);
    PetscReal velocity[3] = {0.0, 0.0, 0.0};  // Initialize velocity array

    // Get phase-specific parameters
    for (std::size_t k = 0; k < phases; k++) {
        Cpk[k] = eosk[k]->GetSpecificHeatCp();
        gammak[k] = eosk[k]->GetSpecificHeatRatio();
        pik[k] = eosk[k]->GetReferencePressure();
    }

    for (std::size_t k = 0; k < phases; k++) {
        alphak[k] = conservedValues[uOff[0] + k];  // alpha_k

        if (alphak[k] > NPhaseFlowFields::ALPHAK_FLOOR) {
            rhok[k] = conservedValues[uOff[1] + k] / alphak[k];  // rho_k = (alpha_k*rho_k)/alpha_k
        } else {
            rhok[k] = 0.0;
            ek[k] = 0.0;
            Tk[k] = 0.0;
            continue;  // Skip further calculations for this phase
        }
    }

    // Compute total density rho = sum_k (alpha_k*rho_k)
    PetscReal rho = 0.0;
    for (std::size_t k = 0; k < phases; k++) {
        rho += conservedValues[uOff[1] + k];  // sum of alpha_k*rho_k
    }

    // Compute velocity components u_i = (rho*u_i)/(rho + PETSC_SMALL)
    PetscReal uiui = 0.0;  // u_i*u_i
    for (PetscInt d = 0; d < dim; d++) {
        velocity[d] = conservedValues[uOff[2] + ablate::finiteVolume::NPhaseFlowFields::RHOU + d] / (rho + PETSC_SMALL);  // u_i = (rho*u_i)/(rho + PETSC_SMALL)
        uiui += velocity[d] * velocity[d];
    }

    // Compute total energy per unit mass e = (rho*e)/(rho + PETSC_SMALL)
    PetscReal e = conservedValues[ablate::finiteVolume::NPhaseFlowFields::RHOE] / (rho + PETSC_SMALL);  // e = (rho*e)/(rho + PETSC_SMALL)

    PetscBool debugThis = PETSC_FALSE;

    // Compute pressure
    PetscReal numerator = conservedValues[uOff[2] + ablate::finiteVolume::NPhaseFlowFields::RHOE];  // rho*e
    numerator -= 0.5 * rho * uiui;  // subtract kinetic energy term

    for (std::size_t k = 0; k < phases; k++) {
      if (alphak[k] < 0) {
        raise(SIGSEGV);
      }
//        if (alphak[k] > NPhaseFlowFields::ALPHAK_FLOOR) {
            numerator -= alphak[k] * gammak[k] * pik[k] / (gammak[k] - 1.0);
//        }
    }

    PetscReal denominator = 0.0;
    for (std::size_t k = 0; k < phases; k++) {
//        if (alphak[k] > NPhaseFlowFields::ALPHAK_FLOOR) {
            denominator += alphak[k] / (gammak[k] - 1.0);
//        }
    }

    // Final pressure calculation
    PetscReal p = numerator / (denominator + PETSC_SMALL);

//if (oldDebugFlag) {
//  printf("%s::%d\t%+e\n", __FILE__, __LINE__, p);
//  exit(0);
//}

    // Debug: Check for problematic pressure values
    if (debugThis) {
        PetscPrintf(MPI_COMM_WORLD, "[DecodeNPhase] Debug: rho=%g, e=%g, uiui=%g, numerator=%g, denominator=%g, p=%g\n",
                    rho, e, uiui, numerator, denominator, p);
    }


    // Compute internal energy per unit mass epsilon = e - u_i*u_i/2
    PetscReal epsilon = e - 0.5 * uiui;

    for (std::size_t k = 0; k < phases; k++) {
        if (alphak[k] > NPhaseFlowFields::ALPHAK_FLOOR) {
            // Compute internal energy per unit mass for phase k
            PetscReal denom_ek = (gammak[k] - 1.0) * rhok[k];
            ek[k] = (p + gammak[k] * pik[k]) / (denom_ek + PETSC_SMALL);

            // Debug: Check for problematic epsilonk values
            if (debugThis) {
                PetscPrintf(MPI_COMM_WORLD, "[DecodeNPhase] Phase %zu: alphak=%g, rhok=%g, p=%g, ek=%g\n",
                            k, alphak[k], rhok[k], p, ek[k]);
            }


            // Compute temperature for phase k
            Tk[k] = gammak[k] * (ek[k] - pik[k]/rhok[k]) / Cpk[k];
            // Compute speed of sound for phase k
            ck[k] = PetscSqrtReal(gammak[k] * (p + pik[k]) / rhok[k]);
            //compute mach number
        } else {
            // Set all phase-specific quantities to zero for zero-phase-fraction phases
            rhok[k] = 0.0;
            ek[k] = 0.0;
            Tk[k] = 0.0;
        }
    }

    // Set output values
    *densityOut = rho;
    *normalVelocityOut = 0.0;
    for (PetscInt d = 0; d < dim; d++) {
        velocityOut[d] = velocity[d];
        *normalVelocityOut += velocity[d] * normal[d];
    }
    *internalEnergyOut = epsilon;

    //*ak = (PetscReal)PetscSqrtReal((gammak - 1.0)*Cpk*Tk);
    //*Mk = normalVelocity / ak;


    for (std::size_t k = 0; k < phases; k++) {
        (*densitykOut)[k] = rhok[k];
        (*internalEnergykOut)[k] = ek[k];
        if (alphak[k] > NPhaseFlowFields::ALPHAK_FLOOR) {
            (*akOut)[k] = PetscSqrtReal((gammak[k] - 1.0) * Cpk[k] * Tk[k]);
            (*MkOut)[k] = *normalVelocityOut / (*akOut)[k];
        } else {
            (*akOut)[k] = 0.0;
            (*MkOut)[k] = 0.0;
        }
        (*TkOut)[k] = Tk[k];
        (*alphakOut)[k] = alphak[k];
    }
    *pOut = p + ALLAIRE_OFFSET*0 + ALPHAK_OFFSET*0 + ALPHAKRHOK_OFFSET*0;

    // Debug print
    // for (std::size_t k = 0; k < phases; k++) {
        // (MPI_COMM_WORLD, "Phase %zu: rhok=%g, ek=%g, Tk=%g, pk=%g, alphak=%g\n",
        //    k, rhok[k], ek[k], Tk[k], p, alphak[k]);
    // }


    //(MPI_COMM_WORLD, "finished DecodeNPhaseAllaireState\n");

}

PetscErrorCode ablate::finiteVolume::processes::NPhaseAllaireAdvectionOld::ZalesakTestSourceTerm(
    PetscInt dim, PetscReal time, const PetscFVCellGeom* cg, const PetscInt uOff[], const PetscScalar u[], const PetscInt aOff[], const PetscScalar a[], PetscScalar f[], void* ctx) {
    PetscFunctionBeginUser;

    auto* nPhase = static_cast<NPhaseAllaireAdvectionOld*>(ctx);
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
REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::NPhaseAllaireAdvectionOld, "", ARG(ablate::eos::EOS, "eos", "must be nPhase"),
         OPT(ablate::parameters::Parameters, "parameters", "the parameters used by advection: cfl(.5)"),
         ARG(ablate::finiteVolume::fluxCalculator::FluxCalculator, "fluxCalculatorNStiff", ""));
