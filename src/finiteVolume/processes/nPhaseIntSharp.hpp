#ifndef ABLATELIBRARY_FINITEVOLUME_INTSHARP_HPP
#define ABLATELIBRARY_FINITEVOLUME_INTSHARP_HPP

#include <petsc.h>
#include <memory>
#include <string>
#include <vector>
#include "domain/range.hpp"
#include "finiteVolume/fluxCalculator/fluxCalculator.hpp"
#include "flowProcess.hpp"
#include "process.hpp"
#include "solver/solver.hpp"
// #include "twoPhaseEulerAdvection.hpp"
#include "nPhaseAllaireAdvection.hpp"

namespace ablate::finiteVolume::processes {

class NPhaseIntSharp : public Process {

   private:
    inline const static std::string FSHARPK_FIELD = "fsharpk";

    std::vector<PetscReal> Gammak;
    std::vector<PetscReal> epsilonk;
    std::vector<PetscInt> flipPhiTildek;
    const bool isPostStep;

    std::shared_ptr<ablate::domain::SubDomain> subDomain;

    // Only necessary for post-step processes
    IS subIS = nullptr;
    Vec subGlobVec = nullptr, subLocVec = nullptr;
    DM subDM = nullptr;
    VecScatter subScatter = nullptr;

   public:

    explicit NPhaseIntSharp(
        const std::vector<PetscReal>& Gammak,
        const std::vector<PetscReal>& epsilonk,
        const std::vector<PetscInt>& flipPhiTildek,
        const bool isPostStep = false);

    ~NPhaseIntSharp() override;

    static PetscErrorCode NPhaseIntSharpPostStep(TS flowTs, ablate::solver::Solver &solver, ablate::finiteVolume::processes::NPhaseIntSharp* process);

    static PetscErrorCode NPhaseIntSharpPreStage(TS ts, ablate::solver::Solver &solver, PetscReal stagetime, ablate::finiteVolume::processes::NPhaseIntSharp* process);

    static PetscErrorCode NPhaseIntSharpPointSource(PetscInt dim, const PetscReal time, const PetscFVCellGeom *cg, const PetscInt *uOff, const PetscScalar *u, const PetscInt *aOff, const PetscScalar *a, PetscScalar *flux, void *ctx);


    void Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) override;
    void Initialize(ablate::finiteVolume::FiniteVolumeSolver &flow) override;


};
}  // namespace ablate::finiteVolume::processes
#endif
