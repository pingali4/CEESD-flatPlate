// Gmsh project created on Mon Apr 26 10:12:21 2021
SetFactory("OpenCASCADE");
Point(1) = {-.2, -0, 0, 1.0};
Point(2) = {0, -0, 0, 1.0};
Point(3) = {1, -0, 0, 1.0};
Point(4) = {1, .1, 0, 1.0};
Point(6) = {-.2, .1, 0, 1.0};
//+
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 6};
Line(6) = {6, 1};
//+
Curve Loop(1) = {1, 2, 3, 4, 6};
Plane Surface(1) = {1};
Physical Curve("inlet", 10) = {6};
Physical Curve("top", 11) = {7};
Physical Curve("plate", 12) = {2};
Physical Curve("bottom", 13) = {1};
Physical Curve("outlet", 14) = {3};
Physical Surface("fluid", 15) = {1};
