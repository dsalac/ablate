#pragma once

#include <petsc.h>
#include <domain/domain.hpp>
#include <domain/range.hpp>
#include <set>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace ablate::finiteVolume {

class SlopeLimiter {
   private:
    std::vector<std::vector<PetscInt>> cellToFaces;

    PetscInt cStart = 0;
    PetscInt cEnd = 0;

    //! flag to indicate if we are in 1D mode
    bool is1D = false;

    //! flag to indicate if we are set up
    bool isSetup = false;

    PetscReal minCellRadius = 0.0;

    std::set<std::string> activatedFields_;

   public:
    SlopeLimiter() = default;
    ~SlopeLimiter() = default;

    bool IsSetup() const { return isSetup; }

    void EnableForField(const std::string& fieldName) { activatedFields_.insert(fieldName); }

    [[nodiscard]] bool IsActiveFor(const std::string& fieldName) const {
        return activatedFields_.find(fieldName) != activatedFields_.end();
    }

    void Setup(DM dm, const domain::Range& cellRange);

    void ApplyLimiter(DM dm, DM dmGrad, PetscInt dim, const domain::Field& field,
                      const domain::Range& cellRange, Vec cellGeomVec, Vec faceGeomVec,
                      const PetscScalar* xLocalArray, PetscScalar* gradGlobArray);
};

}  // namespace ablate::finiteVolume
