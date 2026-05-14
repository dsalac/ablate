#ifndef ABLATELIBRARY_FLUXCALCULATOR_HPP
#define ABLATELIBRARY_FLUXCALCULATOR_HPP
#include <petsc.h>
#include <vector>

namespace ablate::finiteVolume::fluxCalculator {


/**
 * This function returns the flow direction
 * > 0 left to right
 * < 0 right to left
 */
enum Direction { LEFT = 1, RIGHT = 2, NA = 0 };
using FluxCalculatorFunction = Direction (*)(void* ctx, PetscReal uL, PetscReal aL, PetscReal rhoL, PetscReal pL, PetscReal uR, PetscReal aR, PetscReal rhoR, PetscReal pR, PetscReal* massFlux,
                                             PetscReal* p12);

using InterfaceValuesFunction = void (*)(void* ctx, PetscReal uL, PetscReal aL, PetscReal rhoL, PetscReal pL, PetscReal uR, PetscReal aR, PetscReal rhoR, PetscReal pR, PetscReal* a12, PetscReal* m12, PetscReal* p12);

class FluxCalculator {

   public:
    FluxCalculator() = default;
    FluxCalculator(FluxCalculator const&) = delete;
    FluxCalculator& operator=(FluxCalculator const&) = delete;
    virtual ~FluxCalculator() = default;

    // Original interface for backward compatibility
    virtual FluxCalculatorFunction GetFluxCalculatorFunction() = 0;
    virtual void* GetFluxCalculatorContext() { return nullptr; }

    // Function that returns normal velocity, mach number, and pressure at a face
    virtual InterfaceValuesFunction GetInterfaceValuesFunction() { return nullptr; }; // By not using = 0 derived class are not required to implement this function

};
}  // namespace ablate::finiteVolume::fluxCalculator
#endif  // ABLATELIBRARY_FLUXCALCULATOR_HPP
