#ifndef ABLATELIBRARY_FINITEVOLUME_INTSHARP_HPP
#define ABLATELIBRARY_FINITEVOLUME_INTSHARP_HPP

#include <petsc.h>
#include <memory>
#include <vector>
#include "domain/range.hpp"
#include "finiteVolume/fluxCalculator/fluxCalculator.hpp"
#include "flowProcess.hpp"
#include "process.hpp"
#include "solver/solver.hpp"
#include "twoPhaseEulerAdvection.hpp"
#include "finiteVolume/stencils/gaussianConvolution.hpp"
#include "finiteVolume/stencils/stencil.hpp"

namespace ablate::finiteVolume::processes {

class IntSharp : public Process, public ablate::utilities::Loggable<IntSharp> {


   private:
    //coeffs
    const PetscReal epsilonFac = 1.0;
    const PetscReal phiRange[2] = {1.e-6, 1.0 - 1.e-6};



//    ablate::finiteVolume::FiniteVolumeSolver &flowSolver;
    void ClearData();
    std::shared_ptr<ablate::finiteVolume::stencil::GaussianConvolution> cellGaussianConv = nullptr;

    struct vecData {
      Vec vec;
      PetscScalar *array;
    };
    std::vector<struct vecData> vecList = {};
    void MemoryHelper(const Vec baseVec, Vec *newVew, PetscScalar **newArray);
    void MemoryHelper(const DM dm, PetscBool isLocalVec, Vec *newVec, PetscScalar **newArray);
    void MemoryHelper();



   public:
    /**
     *
     * @param Gamma
     * @param epsilon
     */
    explicit IntSharp(const PetscReal epsilonFac = 1);

    /**
     * Clean up the dm created
     */
    ~IntSharp() override;

    /**
     * Setup the process to define the vertex dm
     * @param flow
     */
    void Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) override;
    void Initialize(ablate::finiteVolume::FiniteVolumeSolver &flow) override;

    PetscErrorCode IntSharpPreStep(TS flowTs, ablate::solver::Solver &flow);
    PetscErrorCode IntSharpPreStage(TS flowTS, ablate::solver::Solver &solver, PetscReal stagetime);



};
}  // namespace ablate::finiteVolume::processes
#endif
