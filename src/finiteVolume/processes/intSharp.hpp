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
    const PetscReal Gamma = 0.0;
    const PetscReal epsilonIn = 0.0;
    PetscReal epsilon = 0.0;
    const PetscReal phiRange[2] = {1.e-6, 1.0 - 1.e-6};
//    ablate::finiteVolume::FiniteVolumeSolver &flowSolver;
    void ClearData();
    std::shared_ptr<ablate::finiteVolume::stencil::GaussianConvolution> faceGaussianConv = nullptr;




    /**
     * Store the interpolant for every face
     */
    std::vector<stencil::Stencil> stencils;




   public:
    /**
     *
     * @param Gamma
     * @param epsilon
     */
    explicit IntSharp(PetscReal Gamma, PetscReal epsilon);

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

    static PetscErrorCode ComputeTerm(const FiniteVolumeSolver &solver, DM dm, PetscReal time, Vec locX, Vec locFVec, void *ctx);


    static PetscErrorCode IntSharpFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar field[],
        const PetscScalar grad[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar aux[],
        const PetscScalar gradAux[], PetscScalar flux[], void* ctx);


};
}  // namespace ablate::finiteVolume::processes
#endif
