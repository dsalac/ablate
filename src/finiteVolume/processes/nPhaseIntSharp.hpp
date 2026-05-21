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
#include "nPhaseAllaireAdvection.hpp"
#include "eos/kthStiffenedGas.hpp"

namespace ablate::finiteVolume::processes {

class NPhaseIntSharp : public Process {

   private:
    PetscReal Gamma;
    PetscReal epsilon;

    std::vector<std::shared_ptr<ablate::eos::KthStiffenedGas>> eosNPhase;
    std::shared_ptr<ablate::domain::SubDomain> subDomain;

   public:

    explicit NPhaseIntSharp(
        const PetscReal Gamma,
        const PetscReal epsilon
      );

    ~NPhaseIntSharp() override;

    static PetscErrorCode NPhaseIntSharpPointFlux(PetscInt dim, const PetscFVFaceGeom* fg,
        const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar field[], const PetscScalar grad[],
        const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar aux[], const PetscScalar gradAux[],
        PetscScalar flux[], void* ctx);


    void Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) override;
    void Initialize(ablate::finiteVolume::FiniteVolumeSolver &flow) override;


};
}  // namespace ablate::finiteVolume::processes
#endif
