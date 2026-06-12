#ifndef ABLATELIBRARY_FINITEVOLUME_PRESSURECORRECTION_HPP
#define ABLATELIBRARY_FINITEVOLUME_PRESSURECORRECTION_HPP

#include <petsc.h>
#include <memory>
#include <string>
#include <vector>
#include "domain/range.hpp"
#include "flowProcess.hpp"
#include "process.hpp"
#include "solver/solver.hpp"
#include "nPhaseAllaireAdvection.hpp"
#include "eos/kthStiffenedGas.hpp"

namespace ablate::finiteVolume::processes {

class PressureCorrection : public Process {

   private:
    const PetscReal minPressure;

    std::vector<std::shared_ptr<ablate::eos::KthStiffenedGas>> eosNPhase;
    std::shared_ptr<ablate::domain::SubDomain> subDomain;

    static PetscErrorCode PreStep(FiniteVolumeSolver &fvSolver, TS ts, PetscReal time, bool initialStage, Vec locX, void *ctx);

    static void PressureCorrectionPostEvaluate(TS ts, ablate::solver::Solver&, void *ctx);




   public:

    explicit PressureCorrection(
        const PetscReal minPressure
      );

    ~PressureCorrection() override;

    void Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) override;
    void Initialize(ablate::finiteVolume::FiniteVolumeSolver &flow) override;


};
}  // namespace ablate::finiteVolume::processes
#endif
