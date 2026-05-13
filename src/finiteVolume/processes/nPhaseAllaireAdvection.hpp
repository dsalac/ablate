#ifndef ABLATELIBRARY_NPHASEALLAIREADVECTION_HPP
#define ABLATELIBRARY_NPHASEALLAIREADVECTION_HPP

#include <petsc.h>
#include "eos/stiffenedGas.hpp"
#include "eos/kthStiffenedGas.hpp"
#include "eos/nPhase.hpp"
#include "finiteVolume/nPhaseFlowFields.hpp"
#include "finiteVolume/fluxCalculator/fluxCalculator.hpp"
#include "process.hpp"

// #include "finiteVolume/process.hpp"
#include <memory>
#include <vector>
#include "domain/range.hpp"
#include "eos/eos.hpp"
#include "parameters/parameters.hpp"
#include "finiteVolume/finiteVolumeSolver.hpp"
#include "domain/field.hpp"
#include "domain/region.hpp"
#include "domain/subDomain.hpp"
#include "utilities/petscUtilities.hpp"
#include "finiteVolume/processes/intSharp.hpp"
#include "finiteVolume/stencils/gaussianConvolution.hpp"

namespace ablate::finiteVolume::processes {

class NPhaseAllaireAdvection : public Process {
   public:

    inline const static std::string ALPHAK = eos::NPhase::ALPHAK;
    inline const static std::string ALPHAKRHOK = eos::NPhase::ALPHAKRHOK;
    inline const static std::string ALLAIRE = NPhaseFlowFields::ALLAIRE_FIELD;

    /**
     * General two phase decoder interface
     */
    class NPhaseDecoder {
        public:
         virtual void DecodeNPhaseAllaireState(DM dm, const PetscReal *centroid, PetscInt dim, const PetscInt *uOff, const PetscReal *conservedValues, const PetscReal *normal, PetscReal *density, PetscReal*densityk,
            PetscReal *normalVelocity, PetscReal *velocity, PetscReal *internalEnergy, PetscReal *internalEnergyk, PetscReal *a, PetscReal *ak,
            PetscReal *Mk, PetscReal *p, PetscReal *Tk) = 0;
         virtual ~NPhaseDecoder() = default;
     };


    struct TimeStepData {
        PetscReal cfl;
        eos::ThermodynamicFunction computeSpeedOfSound;
    };
    TimeStepData timeStepData;


   private:

    // This just ensures the proper offset information from uOff is used.
    const std::vector<std::string> conservedFields = {ALPHAK, ALPHAKRHOK, ALLAIRE};
    static const int ALPHAK_FIELD = 0;
    static const int ALPHAKRHOK_FIELD = 1;
    static const int ALLAIRE_FIELD = 2;


    DM subDM;

    PetscErrorCode MultiphaseFlowPreStage(TS flowTs, ablate::solver::Solver &flow, PetscReal stagetime);

    /**
     * Normalize and cleanup the mass fractions in the solution vector
     * @param ts
     */
    static void MultiphaseFlowPostEvaluate(TS ts, ablate::solver::Solver&);






    /**
     * Implementation for two stiffened gases
     */
    class NStiffDecoder : public NPhaseDecoder {

        const std::vector<std::shared_ptr<eos::KthStiffenedGas>> eosk;


        /**
         * Store a scratch euler field for use with the eos
         */
        std::vector<std::vector<PetscReal>> kAllaireFieldScratch;

        /**
         * Get the compute functions using a fake field with only euler
         */
        std::vector<eos::ThermodynamicFunction> kComputeTemperature;
        std::vector<eos::ThermodynamicFunction> kComputeInternalEnergy;
        std::vector<eos::ThermodynamicFunction> kComputeSpeedOfSound;
        std::vector<eos::ThermodynamicFunction> kComputePressure;

       public:
        NStiffDecoder(PetscInt dim, const std::vector<std::shared_ptr<eos::KthStiffenedGas>> &eosk);

        void DecodeNPhaseAllaireState(DM dm, const PetscReal *centroid, PetscInt dim, const PetscInt *uOff, const PetscReal *conservedValues, const PetscReal *normal, PetscReal *density, PetscReal *densityk,
            PetscReal *normalVelocity, PetscReal *velocity, PetscReal *internalEnergy, PetscReal *internalEnergyk, PetscReal *a, PetscReal *ak,
            PetscReal *Mk, PetscReal *p, PetscReal *Tk) override;
    };

    const std::shared_ptr<eos::EOS> eosNPhase;
    std::vector<std::shared_ptr<eos::EOS>> eosk;
    const std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorNStiff;

    // Zalesak test parameters
    bool zalesakTest;
    // PetscReal T_zalesak;

    /**
     * Create and store the decoder
     */
    std::shared_ptr<NPhaseDecoder> decoder;

    std::vector<std::string> auxUpdateFields = {};

   public:

    static PetscErrorCode UpdateAuxFieldsNPhase(PetscReal time, PetscInt dim, const PetscFVCellGeom *cellGeom, const PetscInt uOff[], const PetscScalar *conservedValues, const PetscInt aOff[],
                                                     PetscScalar *auxField, void *ctx);



    NPhaseAllaireAdvection(std::shared_ptr<eos::EOS> eosNPhase, const std::shared_ptr<parameters::Parameters> &parameters,
                           std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorNStiff);
    ~NPhaseAllaireAdvection();
    void Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) override;

   private:
    // static function to compute time step for twoPhase euler advection
    static double ComputeCflTimeStep(TS ts, ablate::finiteVolume::FiniteVolumeSolver &flow, void *ctx);

    static PetscErrorCode NPhaseFlowComputeNPhaseFlux(PetscInt dim, const PetscFVFaceGeom *fg, const PetscInt uOff[], const PetscScalar fieldL[], const PetscScalar fieldR[],
                                                           const PetscInt aOff[], const PetscScalar auxL[], const PetscScalar auxR[], PetscScalar *flux, void *ctx);

//    static PetscErrorCode NPhaseFlowComputeAllaireFlux(PetscInt dim, const PetscFVFaceGeom *fg, const PetscInt uOff[], const PetscScalar fieldL[], const PetscScalar fieldR[],
//                                                           const PetscInt aOff[], const PetscScalar auxL[], const PetscScalar auxR[], PetscScalar *flux, void *ctx);
//    static PetscErrorCode NPhaseFlowComputeAlphakRhokFlux(PetscInt dim, const PetscFVFaceGeom *fg, const PetscInt uOff[], const PetscScalar fieldL[], const PetscScalar fieldR[], const PetscInt aOff[],
//                                                        const PetscScalar auxL[], const PetscScalar auxR[], PetscScalar *flux, void *ctx);
//    static PetscErrorCode NPhaseFlowComputeAlphakFlux(PetscInt dim, const PetscFVFaceGeom *fg, const PetscInt uOff[], const PetscScalar fieldL[], const PetscScalar fieldR[], const PetscInt aOff[],
//                                                        const PetscScalar auxL[], const PetscScalar auxR[], PetscScalar *flux, void *ctx);

    // Zalesak test source term
    static PetscErrorCode ZalesakTestSourceTerm(PetscInt dim, PetscReal time, const PetscFVCellGeom* cg, const PetscInt uOff[], const PetscScalar u[], const PetscInt aOff[], const PetscScalar a[], PetscScalar f[], void* ctx);

    //this will be necessary later
    // static PetscErrorCode NPhaseFlowComputeAlphakFlux(PetscInt dim, const PetscFVFaceGeom *fg, const PetscInt uOff[], const PetscScalar fieldL[], const PetscScalar fieldR[], const PetscInt aOff[],
    //                                                     const PetscScalar auxL[], const PetscScalar auxR[], PetscScalar *flux, void *ctx);

    // Compute the Euler and density-volume fraction fluxes
    static PetscErrorCode NPhaseFlowCompleteFlux(const ablate::finiteVolume::FiniteVolumeSolver &flow, DM dm, PetscReal time, Vec locXVec, Vec locFVec, void* ctx);

   public:
    /**
     * static call to create a NPhaseDecoder based upon eos
     * @param dim
     * @param eosk
     * @return
     */
    static std::shared_ptr<NPhaseDecoder> CreateNPhaseDecoder(PetscInt dim, const std::vector<std::shared_ptr<eos::EOS>> &eosk);
};

}  // namespace ablate::finiteVolume::processes
#endif  // ABLATELIBRARY_NPHASEALLAIREADVECTION_HPP
