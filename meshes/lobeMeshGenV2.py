# this simple python script is used to generate a 2D hex mesh for the specified slab burner geometry
import math
import sys

import gmsh
import argparse


# function to convert the specified locations to gMsh points
def convert_to_point(locations):
    if type(locations) is list:
        points = []
        for location in locations:
            points.append(gmsh.model.geo.add_point(location[0], location[1], 0.0))
        return points
    else:
        return gmsh.model.geo.add_point(locations[0], locations[1], 0.0)


# the sideList is a list of sides
def define_boundary(sides, name, boundary_list):
    line_ids = []
    # march over and add each side
    for side in sides:
        line_ids.append(gmsh.model.geo.add_bspline(side))

    # define the boundary condition for this
    tag_id = gmsh.model.geo.addPhysicalGroup(1, line_ids)
    gmsh.model.setPhysicalName(1, tag_id, name)

    if boundary_list is not None:
        boundary_list.extend(line_ids)


# Initialize gmsh:
gmsh.initialize()

# Define the slab location
beginRamp = 0.05
rampHeight = 0.01
slabLength = 0.07
endSlab = beginRamp + rampHeight + slabLength
endDomain = endSlab + beginRamp
heightMultiplier = 4


# define the experimental chamber points.
lowerLeftPt = convert_to_point((0.0, 0.0))
beginRampPt = convert_to_point((beginRamp, 0.0));
endRampPt = convert_to_point((beginRamp + rampHeight, rampHeight));
endSlabPt = convert_to_point((endSlab, rampHeight));
bottomSlabPt = convert_to_point((endSlab, 0.0));
lowerRightPt = convert_to_point((endDomain, 0.0));
upperRightPt = convert_to_point((endDomain, heightMultiplier*rampHeight));
upperLeftPt = convert_to_point((0, heightMultiplier*rampHeight));



# define the chamber boundary with associated names, define the nodes in a counterclockwise order
boundary_ids = []
define_boundary([[upperLeftPt, lowerLeftPt]], "inlet", boundary_ids)
define_boundary([
  [lowerLeftPt, beginRampPt],
  [beginRampPt, endRampPt],
  [endRampPt, endSlabPt],
  [endSlabPt, bottomSlabPt],
  [bottomSlabPt, lowerRightPt]
], "lower", boundary_ids)
define_boundary([[lowerRightPt, upperRightPt]], "outlet", boundary_ids)
define_boundary([[upperRightPt, upperLeftPt]], "upper", boundary_ids)

# define the curve and resulting plane
curve_id = gmsh.model.geo.add_curve_loop(boundary_ids, reorient=True)
surface_id = gmsh.model.geo.add_plane_surface([curve_id])
gmsh.model.setPhysicalName(2, gmsh.model.geo.addPhysicalGroup(2, [surface_id]), "main")

# Create the relevant Gmsh data structures from Gmsh model.
gmsh.model.geo.synchronize()

# set the default properties to generate quad mesh
gmsh.option.setNumber("Mesh.Algorithm", 11)  # 11: Quasi-structured Quad
gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # 1: Delaunay
gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 3)  # 3: blossom full-quad
gmsh.option.setNumber("Mesh.MeshSizeMin", 0.003)
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.0035)  # with the other options this results in about 0.6 mm element size
gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)  # 1: all quadrangles
gmsh.option.setNumber("Mesh.RecombineAll", 1)  # true

# set the options to prevent gmsh from adding too many elements to each geometry line
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
#gmsh.option.setNumber("Mesh.Smoothing", 10)

# generate the mesh
gmsh.model.mesh.setRecombine(2, surface_id)
gmsh.model.mesh.generate(2)

# parse input arguments
parser = argparse.ArgumentParser(
    description='Generates 2D slabBurner Hex Mesh')
parser.add_argument('--preview', dest='preview', action='store_true',
                    help='If true, preview mesh instead of saving', default=False)
parser.add_argument('--summary', dest='summary', action='store_true',
                    help='If true, computes the element summary', default=False)
args = parser.parse_args()

if args.summary:
    # print a summary of mesh information
    elements = gmsh.model.mesh.getElements(2)
    print(f'Number Elements: {len(elements[1][0])}')
    minDistance = 1E30
    maxDistance = 0
    for ele_tag in elements[1][0]:
        element = gmsh.model.mesh.getElement(ele_tag)
        node_ids = element[1]
        number_nodes = len(node_ids)
        for n in range(number_nodes):
            node_n = gmsh.model.mesh.get_node(node_ids[n])[0]
            for nn in range(n + 1, number_nodes):
                node_nn = gmsh.model.mesh.get_node(node_ids[nn])[0]
                distance = math.sqrt(
                    (node_n[0] - node_nn[0]) ** 2 + (node_n[1] - node_nn[1]) ** 2 + (node_n[2] - node_nn[2]) ** 2)
                minDistance = min(minDistance, distance)
                maxDistance = max(maxDistance, distance)
    print(f'Min/Max Distance: {minDistance}/{maxDistance}')

if args.preview:
    # Creates  graphical user interface
    if 'close' not in sys.argv:
        gmsh.fltk.run()
else:
    # # Write mesh data:
    gmsh.write("lobe2DMeshV2.msh")

# It finalizes the Gmsh API
gmsh.finalize()
