# ***************************************************
# ***** 3D coronary artery model segmentation ******
# ***************************************************
# ***************************************************
# This script contains the functions used in 
# 3D_segmentation_average_property software.
#
# Having the 3D model from CCTA and transformed model
# from CFD simulation, this code transforms the CFD 
# model so that it conforms with the CCTA images.
# The algorithm used is based on Kabsch method.
# Then, having the point IDs of centerline in lesion
# part of artery, this code return the segmented 3D
# model and calculates the average of specified 
# property on each quadrant of each segment.
# ***************************************************
# Developer: Mostafa Mahmoudi
# Start of project: Jan 24, 2021
# Last edit: Mar 7, 2021
# ***************************************************


import numpy
import sys
import vtk
import glob
import re
import math
from vtk.util import numpy_support as VN
from vmtk import pypes
from vmtk import vmtkscripts
import re
from scipy.spatial.transform import Rotation as Rot


# ***************************************************
# ******* Extract the centerline using VMTK *********
# ***************************************************

def extract_centerline_vmtk(file_name, output_root):
		global output_centerline

		# CL_split = 'vmtkcenterlines -seedselector openprofiles -ifile %s -ofile %scenterlines.vtp --pipe vmtkbranchextractor -ofile %scenterline_split.vtp --pipe vmtksurfacereader -ifile %s --pipe vmtkbranchclipper -groupids -ofile %sfoo_sp_model.vtp --pipe vmtksurfaceviewer -legend 1 -array GroupIds' % (file_name, output_root,output_root,file_name, output_root)

		CL_split = 'vmtkcenterlines -seedselector openprofiles -ifile %s -ofile %scenterlines.vtp --pipe vmtkbranchextractor -ofile %scenterline_split.vtp --pipe vmtksurfacereader -ifile %s --pipe vmtkbranchclipper -groupids -ofile %sfoo_sp_model.vtp --pipe vmtksurfaceviewer -legend 1 -array GroupIds' % (file_name, output_root,output_root,file_name, output_root)


		myPype = pypes.PypeRun(CL_split)

		print('Centerline successfully was created.\n Centerline File address: %scenterlines.vtp' %output_root)

		output_centerline = '%scenterline_split.vtp' %output_root

		output_centerline_vtk = myPype.GetScriptObject('vmtkbranchextractor','0').Centerlines

		print("Output centerline type: ", type(output_centerline_vtk))

		return output_centerline, output_centerline_vtk



# ***************************************************
# ******* Extract the target centerline branch ******
# ******* using VMTK						   ******
# ***************************************************


def extract_centerline_branch(input_centerline):

		# input args:
		# input_centerline: vtkPolyData - The output from vmtkbranchextractor

		global CL_points

		surfaceViewer = vmtkscripts.vmtksurfaceviewer()
		surfaceViewer.Surface = input_centerline
		surfaceViewer.ArrayName = "CenterlineIds"
		surfaceViewer.Legend = 1
		surfaceViewer.DisplayCellData = 1
		surfaceViewer.Execute()

		print('*****************************************************')
		print('User Input Required!')
		print('*****************************************************')

		centerline_ID = input("Enter the target branch ID: ")





		print('Loading', file_name)
		reader = vtk.vtkXMLPolyDataReader()
		reader.SetFileName(file_name)
		reader.Update()
		data_CL = reader.GetOutput()
		print(data_CL)
		n_points_CL = data_CL.GetNumberOfPoints()

		points = data_CL.GetPoints()
		point_id = points.GetPoint(2) # 2 is the point id

		num_cells = data_CL.GetNumberOfCells()
		print('NUMBER OF CELLS: ', num_cells)

		polyline = data_CL.GetCell(0) # 0 is the cell id which can be identified from the last figure in VMTK interactive environment
		n_points_polyline = polyline.GetNumberOfPoints()
		print('number of points in the polyline:', n_points_polyline)

		# test_point = polyline.GetPointId(n_points_polyline)
		# print('Last centerline Point: ', data_CL.GetPoint(test_point))

		# Get all points on the polyline based on the point ID
		CL_points = numpy.zeros((n_points_polyline, 3))
		idx = 0
		for i in range(polyline.GetNumberOfPoints()):
			pointId = polyline.GetPointId(i) # pointId is the point id of the i-th point along the polyline
			CL_points[idx] = data_CL.GetPoint(pointId)
			idx += 1

		print(CL_points)
		print(numpy.linalg.norm(CL_points[1] - CL_points[0]))
		# WSS = VN.vtk_to_numpy(data_CL.GetCell(1).GetNumberOfArrays())
		# print(WSS)
		return CL_points


# ***************************************************
# *************** Visualize the data ****************
# ***************************************************

def render_view(output_port):

		# vtkPolyDataMapper is a class that maps polygonal data (i.e., vtkPolyData) to 
		# graphics primitives. vtkPolyDataMapper serves as a superclass for 
		# device-specific poly data mappers, that actually do the mapping to the 
		# rendering/graphics hardware/software.

		# global picked_point

		wss_cell_array = output_port.GetCellData().GetArray('Blanking')
		print(wss_cell_array)

		lut = vtk.vtkLookupTable()

		mapper = vtk.vtkPolyDataMapper()
		# mapper = vtk.vtkDataSetMapper()
		mapper.SetInputData(output_port)
		mapper.SetScalarModeToUseCellFieldData();
		mapper.SetColorModeToMapScalars()
		mapper.SetUseLookupTableScalarRange(True)
		mapper.ScalarVisibilityOn();
		# mapper.SetLookupTable(lut)
		mapper.SetScalarRange(output_port.GetCellData().GetArray("Blanking").GetRange());
		mapper.SelectColorArray("Blanking");
		print(output_port.GetCellData().GetArray("Blanking"))

		mapper.Update()

		

		


		# vtkActor is used to represent an entity in a rendering scene.
		actor = vtk.vtkActor()
		actor.SetMapper(mapper)

		# vtkRenderer provides an abstract specification for renderers. A renderer is an 
		# object that controls the rendering process for objects. Rendering is the process 
		# of converting geometry, a specification for lights, and a camera view into an image.
		renderer = vtk.vtkRenderer()
		renderer.AddActor(actor)
		renderer.SetBackground(0.1, 0.2, 0.4)

		# create a window for renderers to draw into
		# vtkRenderWindow is an abstract object to specify the behavior of a rendering window. 
		# A rendering window is a window in a graphical user interface where renderers draw their images.
		render_window = vtk.vtkRenderWindow()
		render_window.AddRenderer(renderer)

		# Automatically set up the camera based on the visible actors.
		renderer.ResetCamera()

		


		interactive_window = vtk.vtkRenderWindowInteractor()
		interactive_window.SetRenderWindow(render_window)

		style = vtk.vtkInteractorStyleTrackballCamera()
		interactive_window.SetInteractorStyle(style)

		# # create the scalar_bar
		# scalar_bar = vtk.vtkScalarBarActor()
		# scalar_bar.SetOrientationToHorizontal()
		# scalar_bar.SetLookupTable(lut)

		# # create the scalar_bar_widget
		# scalar_bar_widget = vtk.vtkScalarBarWidget()
		# scalar_bar_widget.SetInteractor(interactive_window)
		# scalar_bar_widget.SetScalarBarActor(scalar_bar)
		# scalar_bar_widget.On()

		render_window.Render()

		picked_coords = []
		# global_p_coords = [0,0,0]
		picked_point = [0,0,0]

		def mark(x,y,z):
			"""mark the picked location with a sphere"""
			global picked_point
			# print(x,y,z)
			sphere = vtk.vtkSphereSource()
			sphere.SetRadius(0.02)
			res = 20
			sphere.SetThetaResolution(res)
			sphere.SetPhiResolution(res)
			sphere.SetCenter(x,y,z)
			sphere.Update()

			mapper_sphere = vtk.vtkPolyDataMapper()
			mapper_sphere.SetInputData(sphere.GetOutput())

			marker = vtk.vtkActor()
			marker.SetMapper(mapper_sphere)

			renderer.AddActor(marker)
			render_window.AddRenderer(renderer)

			marker.GetProperty().SetColor( (1,0,0) )
			interactive_window.Render()
			render_window.Render()

			picked_point = [x,y,z]
			print('test1: ', picked_point)
			
			return picked_point



		def pick_cell(renwinInteractor, event):
			global picked_point

			x, y = renwinInteractor.GetEventPosition()

			picker = vtk.vtkCellPicker()
			picker.SetTolerance(0.0005)
			picker.Pick(x, y, 0, renderer)
			vid =  picker.GetCellId()
			pid = picker.GetPointId()
			if vid!=-1:
			    sid = picker.GetSubId()
			    pcoords = picker.GetPCoords()
			    global_p_coords = output_port.GetPoint(pid)
			    print(vid, sid, pid, pcoords, global_p_coords)
			    dataSet = picker.GetDataSet()
			    pnt = dataSet.GetPoint(vid)
			    picked_point = mark(*global_p_coords)
			    print('test2: ', picked_point)
			else:
			    print('swing and a miss')


		def keypress_callback(obj, ev):
			global picked_point
			key = obj.GetKeySym()
			if key == 'p':
				picked_coords.append(picked_point)
				print('Picked Point: ', picked_point)
				sphere = vtk.vtkSphereSource()
				sphere.SetRadius(0.025)
				res = 20
				sphere.SetThetaResolution(res)
				sphere.SetPhiResolution(res)
				sphere.SetCenter(picked_point[0],picked_point[1],picked_point[2])
				sphere.Update()

				mapper_sphere = vtk.vtkPolyDataMapper()
				mapper_sphere.SetInputData(sphere.GetOutput())

				marker = vtk.vtkActor()
				marker.SetMapper(mapper_sphere)

				renderer.AddActor(marker)
				render_window.AddRenderer(renderer)

				marker.GetProperty().SetColor( (0,1,0) )
				# interactive_window.Render()
				render_window.Render()

			if key == 'Return':
				final_render_window = interactive_window.GetRenderWindow()
				final_render_window.Finalize()
				interactive_window.TerminateApp()
				del final_render_window, interactive_window


		interactive_window.AddObserver('LeftButtonPressEvent', pick_cell)

		interactive_window.AddObserver('KeyPressEvent', keypress_callback)

		interactive_window.Initialize()
		
		interactive_window.Start()

		print(picked_coords)
		return picked_coords


# ***************************************************
# ******* Find rotation and translation *************
# ******* matrices using Kabsch method  *************
# ***************************************************

def Kabsch_transformation(A,B):

		assert A.shape == B.shape

		num_rows, num_cols = A.shape
		if num_rows != 3:
		    raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

		num_rows, num_cols = B.shape
		if num_rows != 3:
		    raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

		# find mean column wise
		centroid_A = numpy.mean(A, axis=1)
		centroid_B = numpy.mean(B, axis=1)

		# ensure centroids are 3x1
		centroid_A = centroid_A.reshape(-1, 1)
		centroid_B = centroid_B.reshape(-1, 1)

		# subtract mean
		Am = A - centroid_A
		Bm = B - centroid_B

		H = Am @ numpy.transpose(Bm)

		# sanity check
		#if linalg.matrix_rank(H) < 3:
		#    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

		# find rotation
		U, S, Vt = numpy.linalg.svd(H)
		R = Vt.T @ U.T

		# special reflection case
		if numpy.linalg.det(R) < 0:
		    print("det(R) < R, reflection detected!, correcting for it ...")
		    Vt[2,:] *= -1
		    R = Vt.T @ U.T

		t = -R @ centroid_A + centroid_B

		return R, t


# ***************************************************
# ******* Construct the transformation matrix *******
# ***************************************************

def Get_Rotation_translation(R, t):

		transformation_matrix = vtk.vtkMatrix4x4()
		transformation_matrix.DeepCopy((R[0][0], R[0][1],R[0][2], t[0],
		                   R[1][0], R[1][1],R[1][2], t[1],
		                   R[2][0], R[2][1],R[2][2], t[2],
		                   0, 0, 0, 1))

		return transformation_matrix


# ***************************************************
# ************** Transform the 3D model *************
# ***************************************************

def transformation_3D_model_matrix(file_name, output_root, trans_matrix):

		# Read the input file
		print('Loading', file_name)
		reader = vtk.vtkXMLPolyDataReader()
		reader.SetFileName(file_name)
		reader.Update()
		print('Done loading input file.')

		print('Performing transformation ...')
		transformation_T1 = vtk.vtkTransform()
		transformation_T1.SetMatrix(trans_matrix)
		transformation_T1.Update()

		transformFilter_T1 = vtk.vtkTransformPolyDataFilter()
		transformFilter_T1.SetInputConnection(reader.GetOutputPort())
		transformFilter_T1.SetTransform(transformation_T1)
		transformFilter_T1.Update()
		print('Transformation Done!')

		myoutput = vtk.vtkDataSetWriter()
		myoutput.SetInputConnection(transformFilter_T1.GetOutputPort())
		myoutput.SetFileName('%stransformed_model_Kabsch3.vtk' %output_root)
		myoutput.Write()

		return transformFilter_T1


def transformation_LCS_model_matrix(file_name, output_root, trans_matrix):

		# Read the input file
		print('Loading', file_name)
		reader = vtk.vtkPolyDataReader()
		reader.SetFileName(file_name)
		reader.Update()
		print('Done loading input file.')

		print('Performing transformation ...')
		transformation_T1 = vtk.vtkTransform()
		transformation_T1.SetMatrix(trans_matrix)
		transformation_T1.Update()

		transformFilter_T1 = vtk.vtkTransformPolyDataFilter()
		transformFilter_T1.SetInputConnection(reader.GetOutputPort())
		transformFilter_T1.SetTransform(transformation_T1)
		transformFilter_T1.Update()
		print('Transformation Done!')

		myoutput = vtk.vtkDataSetWriter()
		myoutput.SetInputConnection(transformFilter_T1.GetOutputPort())
		myoutput.SetFileName('%stransformed_LCS_Kabsch3.vtk' %output_root)
		myoutput.Write()

		return transformFilter_T1


# ***************************************************
# ********* Extract CL coords from SV .pth **********
# ***************************************************

def read_SV_centerline(filename):
		infile = open(filename, 'r') 
		numbers_coords = []
		numbers_tangent = []

		lines = infile.readlines()

		for line in lines:
			if 'pos x' in line:
				numbers_str = numpy.array(re.findall('[\+\-]?\d*\.?\d+',line))
				numbers_tmp = numbers_str.astype(numpy.float)
				numbers_coords.append(numbers_tmp.tolist())

			if 'tangent x' in line:
				numbers_str = numpy.array(re.findall('[\+\-]?\d*\.?\d+',line))
				numbers_tmp = numbers_str.astype(numpy.float)
				numbers_tangent.append(numbers_tmp.tolist())

		return numbers_coords, numbers_tangent


# ***************************************************
# ********* Calculate the tangents manually *********
# ***************************************************

def create_tangents(centerline):

		tangent_vector = []
		number_of_points = len(centerline)

		for idx in range(number_of_points):
			if idx == (number_of_points - 1):
				tangent_vector.append(tangent_vector[idx - 1])
			else:
				tangent_vector.append( ( numpy.subtract(centerline[idx+1], centerline[idx])/numpy.linalg.norm( numpy.subtract(centerline[idx+1], centerline[idx]) ) ).tolist() )

		# print(tangent_vector)
		return tangent_vector


def update_tangents(tangents, CL_size):

		tangents_updated = []
		number_of_points = len(tangents)

		for idx in range(number_of_points):
			if idx == (number_of_points - 1):
				tangents_updated.append(tangents[idx - 1])
			else:
				tangents_updated.append( (numpy.add(tangents[idx], tangents[idx+1])/numpy.linalg.norm( numpy.add(tangents[idx], tangents[idx+1]) ) ).tolist() )

		print("tangents_updated : ", len(tangents_updated))

		tangent_aug = []
		tangent_counter = 0
		for idx in range(CL_size):
			print(tangent_counter)
			print("idx: ", idx)
			if idx >= (CL_size - 5):
				tangent_aug.append(tangents_updated[len(tangents_updated)-1])
			else:
				tangent_aug.append(tangents_updated[tangent_counter])
				if (idx % 4 == 0):
					tangent_counter += 1


		print('***************** augmented tangents *******************')
		print(len(tangent_aug))
		print(tangent_aug)


		# return tangents_updated
		return tangent_aug


# ***************************************************
# ********** Create segments with sectors ***********
# ***************************************************

def create_segment_locations(model_vtkXMLPolyData, centerline, tangents, slice_id, fieldname, output_root):

		# Read the input file
		# print('Loading', file_name)
		# reader = vtk.vtkXMLPolyDataReader()
		# reader.SetFileName(file_name)
		# reader.Update()
		# data_wss = reader.GetOutput()
		# n_points = data_wss.GetNumberOfPoints()

		plane_start = vtk.vtkPlane()
		plane_start.SetOrigin(centerline[0][0], centerline[0][1], centerline[0][2])
		plane_start.SetNormal(tangents[0][0], tangents[0][1], tangents[0][2])

		plane_end = vtk.vtkPlane()
		plane_end.SetOrigin(centerline[1][0], centerline[1][1], centerline[1][2])
		plane_end.SetNormal(tangents[0][0], tangents[0][1], tangents[0][2])

		clipper_start = vtk.vtkClipPolyData()
		clipper_start.SetInputConnection(model_vtkXMLPolyData.GetOutputPort())
		clipper_start.SetClipFunction(plane_start)
		clipper_start.InsideOutOff()
		clipper_start.Update()
		# data_angle_clipped = clipper.GetOutputPort()

		clipper_end = vtk.vtkClipPolyData()
		clipper_end.SetInputConnection(clipper_start.GetOutputPort())
		clipper_end.SetClipFunction(plane_end)
		clipper_end.InsideOutOn()
		clipper_end.Update()

		myoutput1 = vtk.vtkDataSetWriter()
		myoutput1.SetInputConnection(clipper_end.GetOutputPort())
		# myoutput1.SetFileName('%sTEST_clipped1.vtk' %output_root)
		myoutput1.SetFileName(output_root+'seg_'+str(slice_id)+'.vtk')
		myoutput1.Write()



		# angles, R = Find_Rotation_angles([0,0,1], tangents[0])

		R = Find_Rotation_Matrix_VTK_style([0,0,1], tangents[0])

		print("Normal = ", tangents[0])
		# print("angles = ", angles)
		print("transformation Matrix = ", R)

		# normal_x = numpy.dot(R, [1,0,0,0])
		# normal_y = numpy.dot(R, [0,1,0,0])
		normal_x = numpy.dot(R, [1,0,0])
		normal_y = numpy.dot(R, [0,1,0])

		print("normal_x = ", normal_x)
		print("normal_y = ", normal_y)

		# plane2 = vtk.vtkPlane()
		# plane2.SetOrigin(centerline[3][0], centerline[3][1], centerline[3][2])
		# plane2.SetNormal(normal_x[0], normal_x[1], normal_x[2])

		# plane3 = vtk.vtkPlane()
		# plane3.SetOrigin(centerline[3][0], centerline[3][1], centerline[3][2])
		# plane3.SetNormal(normal_y[0], normal_y[1], normal_y[2])
		# center_point = [centerline[3][0], centerline[3][1], centerline[3][2]]

		sector1, sector4, sector3, sector2 = create_sectors(clipper_end, centerline[0], normal_x, normal_y)

		# create_sectors(data_angle_clipped, center_point, normal_x, normal_y)

		print("All sectors are created successfully!")

		write_sectors(sector1, sector2, sector3, sector4, slice_id, output_root)

		sec1_average_value = compute_average_surface_value(sector1, fieldname)
		sec2_average_value = compute_average_surface_value(sector2, fieldname)
		sec3_average_value = compute_average_surface_value(sector3, fieldname)
		sec4_average_value = compute_average_surface_value(sector4, fieldname)

		return sec1_average_value, sec2_average_value, sec3_average_value, sec4_average_value


# ***************************************************
# ***************** Create sectors ******************
# ***************************************************

def create_sectors(segment, center_point, normal_x, normal_y):

		plane1 = vtk.vtkPlane()
		plane1.SetOrigin(center_point[0], center_point[1], center_point[2])
		plane1.SetNormal(normal_x[0], normal_x[1], normal_x[2])

		plane2 = vtk.vtkPlane()
		plane2.SetOrigin(center_point[0], center_point[1], center_point[2])
		plane2.SetNormal(normal_y[0], normal_y[1], normal_y[2])

		# Sector #1
		sector_tmp1 = vtk.vtkClipPolyData()
		sector_tmp1.SetInputConnection(segment.GetOutputPort())
		sector_tmp1.SetClipFunction(plane1)
		sector_tmp1.InsideOutOff()
		sector_tmp1.Update()
		sector_tmp1_out = sector_tmp1.GetOutputPort()

		sector1 = vtk.vtkClipPolyData()
		sector1.SetInputConnection(sector_tmp1_out)
		sector1.SetClipFunction(plane2)
		sector1.InsideOutOn()
		sector1.Update()
		sector1_outputPort = sector1.GetOutputPort()

		print("Sec1 Created!")

		# Sector #2
		sector_tmp2 = vtk.vtkClipPolyData()
		sector_tmp2.SetInputConnection(segment.GetOutputPort())
		sector_tmp2.SetClipFunction(plane1)
		sector_tmp2.InsideOutOn()
		sector_tmp2.Update()
		sector_tmp2_out = sector_tmp2.GetOutputPort()

		sector2 = vtk.vtkClipPolyData()
		sector2.SetInputConnection(sector_tmp2_out)
		sector2.SetClipFunction(plane2)
		sector2.InsideOutOn()
		sector2.Update()
		sector2_outputPort = sector2.GetOutputPort()

		print("Sec2 Created!")

		# Sector #3
		sector_tmp3 = vtk.vtkClipPolyData()
		sector_tmp3.SetInputConnection(segment.GetOutputPort())
		sector_tmp3.SetClipFunction(plane1)
		sector_tmp3.InsideOutOn()
		sector_tmp3.Update()
		sector_tmp3_out = sector_tmp3.GetOutputPort()

		sector3 = vtk.vtkClipPolyData()
		sector3.SetInputConnection(sector_tmp3_out)
		sector3.SetClipFunction(plane2)
		sector3.InsideOutOff()
		sector3.Update()
		sector3_outputPort = sector3.GetOutputPort()

		print("Sec3 Created!")

		# Sector #4
		sector_tmp4 = vtk.vtkClipPolyData()
		sector_tmp4.SetInputConnection(segment.GetOutputPort())
		sector_tmp4.SetClipFunction(plane1)
		sector_tmp4.InsideOutOff()
		sector_tmp4.Update()
		sector_tmp4_out = sector_tmp4.GetOutputPort()

		sector4 = vtk.vtkClipPolyData()
		sector4.SetInputConnection(sector_tmp4_out)
		sector4.SetClipFunction(plane2)
		sector4.InsideOutOff()
		sector4.Update()
		sector4_outputPort = sector4.GetOutputPort()

		print("Sec4 Created!")

		# return sector1_outputPort, sector2_outputPort, sector3_outputPort, sector4_outputPort
		return sector1, sector2, sector3, sector4


# ***************************************************
# ************** Write sectors as .vtk **************
# ***************************************************

def write_sectors(sec1, sec2, sec3, sec4, slice_id, output_root):

		print("Start writing sector 1 ...")

		myoutput1 = vtk.vtkDataSetWriter()
		myoutput1.SetInputConnection(sec1.GetOutputPort())
		# myoutput1.SetFileName('%ssec_1.vtk' %output_root)
		myoutput1.SetFileName(output_root+'seg_'+str(slice_id)+'sec_1'+'.vtk')
		myoutput1.Write()

		print("Start writing sector 2 ...")

		myoutput2 = vtk.vtkDataSetWriter()
		myoutput2.SetInputConnection(sec2.GetOutputPort())
		# myoutput2.SetFileName('%ssec_2.vtk' %output_root)
		myoutput2.SetFileName(output_root+'seg_'+str(slice_id)+'sec_2'+'.vtk')
		myoutput2.Write()

		print("Start writing sector 3 ...")

		myoutput3 = vtk.vtkDataSetWriter()
		myoutput3.SetInputConnection(sec3.GetOutputPort())
		# myoutput3.SetFileName('%ssec_3.vtk' %output_root)
		myoutput3.SetFileName(output_root+'seg_'+str(slice_id)+'sec_3'+'.vtk')
		myoutput3.Write()

		print("Start writing sector 4 ...")

		myoutput4 = vtk.vtkDataSetWriter()
		myoutput4.SetInputConnection(sec4.GetOutputPort())
		# myoutput4.SetFileName('%ssec_4.vtk' %output_root)
		myoutput4.SetFileName(output_root+'seg_'+str(slice_id)+'sec_4'+'.vtk')
		myoutput4.Write()


# ***************************************************
# *********** Compute the surface average ***********
# ***************************************************

def compute_average_surface_value(segment, fieldname):

		data = segment.GetOutput()
		wss  = data.GetPointData().GetArray(fieldname)
		wss_array = VN.vtk_to_numpy(wss) #change format from vtk to numpy

		qualityFilter = vtk.vtkMeshQuality()
		qualityFilter.SetInputData(data)
		qualityFilter.SetTriangleQualityMeasureToArea()
		qualityFilter.Update()

		qualityMesh = qualityFilter.GetOutput()
		qualityArray = qualityMesh.GetCellData().GetArray("Quality")
		# print(qualityArray)
		qualityArray_np = VN.vtk_to_numpy(qualityArray)
		# print(qualityArray_np)
		# print("quality size = ", qualityArray_np.size)
		# print("quality shape = ", qualityArray_np.shape)


		# print("Quality calculation DONE!")

		p2c = vtk.vtkPointDataToCellData()
		p2c.SetInputData(data)
		p2c.PassPointDataOn()
		p2c.Update()
		wss_cell = p2c.GetOutput()

		wss_cell_array = wss_cell.GetCellData().GetArray(fieldname)
		wss_cell_array_np = VN.vtk_to_numpy(wss_cell_array)

		# print(wss_cell_array_np)
		# print("wss size = ", wss_cell_array_np.size)
		# print("wss shape = ", wss_cell_array_np.shape)

		WSS_ave = numpy.zeros(wss_cell.GetNumberOfCells())
		WSS_ds = 0.0
		Area = 0.0

		cell_number = wss_cell.GetNumberOfCells()
		for cell_id in range(cell_number):
			WSS_ave[cell_id] = numpy.linalg.norm(wss_cell_array_np[cell_id])

			WSS_ds += WSS_ave[cell_id]*qualityArray_np[cell_id]
			Area += qualityArray_np[cell_id]

		WSS_area_average = WSS_ds/Area

		print("Area-weighted WSS average = ", WSS_area_average)

		return WSS_area_average


def Find_Rotation_angles(initial_vector, target_vector):
		dot_product = numpy.dot(initial_vector, target_vector)
		cross_product = numpy.cross(initial_vector, target_vector)

		v_skew = numpy.zeros((3, 3))

		v_skew[0][0] = 0.0
		v_skew[0][1] = -cross_product[2]
		v_skew[0][2] = cross_product[1]

		v_skew[1][0] = cross_product[2]
		v_skew[1][1] = 0.0
		v_skew[1][2] = -cross_product[0]

		v_skew[2][0] = -cross_product[1]
		v_skew[2][1] = cross_product[0]
		v_skew[0][1] = 0.0

		rotation_matrix = numpy.identity(3) + v_skew + numpy.matmul(v_skew, v_skew) * (1/(1 + dot_product))

		# Finding the Euler angles

		alpha = numpy.arctan2(rotation_matrix[2][1], rotation_matrix[2][2])
		beta = numpy.arctan2(-rotation_matrix[2][0], numpy.sqrt(rotation_matrix[2][1]**2 + rotation_matrix[2][2]**2) )
		gamma = numpy.arctan2(rotation_matrix[1][0], rotation_matrix[0][0]) 

		alpha_ang = numpy.arctan2(rotation_matrix[2][1], rotation_matrix[2][2]) * 180/numpy.pi
		beta_ang = numpy.arctan2(-rotation_matrix[2][0], numpy.sqrt(rotation_matrix[2][1]**2 + rotation_matrix[2][2]**2) ) * 180/numpy.pi
		gamma_ang = numpy.arctan2(rotation_matrix[1][0], rotation_matrix[0][0]) * 180/numpy.pi

		angles = [alpha_ang, beta_ang, gamma_ang]

		print("alpha = % 5.4f, beta = % 5.4f, gamma = % 5.2f" %(alpha_ang, beta_ang, gamma_ang))

		rot_matrix_x = [[1, 0, 0, 0], [0, numpy.cos(alpha), -numpy.sin(alpha), 0], [0, numpy.sin(alpha), numpy.cos(alpha), 0], [0, 0, 0, 1]]
		rot_matrix_y = [[numpy.cos(beta), 0, numpy.sin(beta), 0], [0, 1, 0, 0], [-numpy.sin(beta), 0, numpy.cos(beta), 0], [0, 0, 0, 1]]
		rot_matrix_z = [[numpy.cos(gamma), -numpy.sin(gamma), 0, 0], [numpy.sin(gamma), numpy.cos(gamma), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

		R = numpy.dot(rot_matrix_z, (numpy.dot(rot_matrix_y, rot_matrix_x)))
		#R = rot_matrix_z(gamma) * rot_matrix_y(beta) * rot_matrix_x(alpha);

		return angles, R


def Find_Rotation_Matrix_VTK_style(initial_vector, target_vector):
	
		theta = numpy.arccos(numpy.dot(initial_vector, target_vector))
		axis_rotation = numpy.cross(initial_vector, target_vector)/numpy.linalg.norm(numpy.cross(initial_vector, target_vector))

		r = Rot.from_rotvec(theta * axis_rotation)

		R = r.as_dcm()

		return R




def write_output_dataset(path, dataset):
    with open(path, "wt") as file:
        for row in dataset:
            for item in row:
                file.write(str(item) + " ")
            file.write("\n")


def point_distribution_along_centerline(points, input_arg, flag_uniformSpacing):

		currentpt = points[0]	# current point
		indfirst = 1			# index of the most closest point in p from curpt
		length = len(points)	# length of points
		q = []
		q.append(currentpt)		# output point
		k = 0;
		dist_between_pts = numpy.zeros(length-1)

		# distance between points in points
		for k0 in range(length-1):
		   dist_between_pts[k0] = distance_3d(points[k0], points[k0+1])

		total_dist = numpy.sum(dist_between_pts);
		
		# Target interval
		if flag_uniformSpacing:
			interval_target = input_arg
			N = int((total_dist/interval_target) + 1)
		else:
		    interval_target = total_dist/(input_arg - 1);



		for k in range(N-1):

		   new_pt = []
		   distsum = 0
		   ptnow = currentpt
		   kk = 0
		   pttarget = points[indfirst]
		   remainder = interval_target	# remainder of distance that should be accumulated
		   while not new_pt:
		      # calculate the distance from active point to the most
		      # closest point in points

		      disttmp = distance_3d(ptnow, pttarget)
		      distsum = distsum + disttmp
		      # if distance is enough, generate newpt. else, accumulate distance

		      if distsum >= interval_target:
		         new_pt = interpintv(ptnow, pttarget, remainder)
		      else:
		         remainder = remainder - disttmp
		         ptnow = pttarget
		         kk = kk + 1;
		         if indfirst+kk > length:
		            new_pt = points[length]
		         else:
		            pttarget = points[indfirst+kk]


		   # add to the output points
		   q.append(new_pt)

		   # update currentpt and indfirst
		   currentpt = new_pt
		   indfirst = indfirst + kk

		return q

def distance_3d(x,y):
    	# calculate distance
		l = numpy.sqrt( (x[0]-y[0])**2+(x[1]-y[1])**2+(x[2]-y[2])**2 )

		return l

def interpintv(pt1,pt2,intv):

    # Generate a point between pt1 and pt2 in such a way that
    # the distance between pt1 and new point is intv.
    # pt1 and pt2 should be 1x3 or 1x2 vector.

	dirvec = numpy.subtract(pt2, pt1)
	dirvec = dirvec/numpy.linalg.norm(dirvec)
	l = dirvec[0]
	m = dirvec[1]
	n = dirvec[2]
	newpt = [ intv*l+pt1[0], intv*m+pt1[1], intv*n+pt1[2] ]

	return newpt


def split_CL_at_bifurcation(CL, bif_point):

	err = numpy.subtract(CL, bif_point)
	err_mag = []

	for idx in range(len(CL)):
		err_mag.append( numpy.linalg.norm(err[idx]) )

	# print("err_mag = ", err_mag)
	# err_min = numpy.amin(err_mag)
	# bif_index = numpy.where(err_mag == numpy.amin(err_mag))
	bif_index = numpy.unravel_index(numpy.argmin(err_mag),numpy.shape(err_mag))[0]

	print("bifurcation index = ", bif_index)

	CL_sp1 = ( numpy.flip(CL[0:bif_index+1], 0) ).tolist()
	print("Before bifurcation: ", len(CL_sp1))
	CL_sp2 = ( CL[bif_index+1:] )
	print("After bifurcation: ", len(CL_sp2))

	CL_sp1_interp = point_distribution_along_centerline(CL_sp1, 0.5, True)
	CL_sp2_interp = point_distribution_along_centerline(CL_sp2, 0.5, True)
	new_bif_idx = len(CL_sp1_interp)
	print("Bifurcation index after redistribution: ", new_bif_idx)
	print("Before bifurcation: ", len(CL_sp1_interp))	
	print("After bifurcation: ", len(CL_sp2_interp))

	CL_splitted = ( numpy.append(numpy.flip( CL_sp1_interp, 0 ), CL_sp2_interp, axis=0) ).tolist()
	print("CL splitted: ", len(CL_splitted))

	# print(CL_splitted)

	return CL_splitted


# ssh -Y aa3878@lrc108-01.egr.nau.edu


def read_CL_VTK(file_name):

	print('Loading', file_name)
	reader = vtk.vtkXMLPolyDataReader()
	reader.SetFileName(file_name)
	reader.Update()
	data_CL = reader.GetOutput()

	CL_cell_array = data_CL.GetCellData().GetArray('CenterlineIds')
	# print(CL_cell_array)

	lut = vtk.vtkLookupTable()

	mapper = vtk.vtkPolyDataMapper()
	# mapper = vtk.vtkDataSetMapper()
	mapper.SetInputData(data_CL)
	mapper.SetScalarModeToUseCellFieldData();
	mapper.SetColorModeToMapScalars()
	mapper.SetUseLookupTableScalarRange(True)
	mapper.ScalarVisibilityOn();
	# mapper.SetLookupTable(lut)
	mapper.SetScalarRange(data_CL.GetCellData().GetArray("CenterlineIds").GetRange());
	mapper.SelectColorArray("CenterlineIds");
	print(data_CL.GetCellData().GetArray("CenterlineIds"))

	mapper.Update()

	# vtkActor is used to represent an entity in a rendering scene.
	actor = vtk.vtkActor()
	actor.SetMapper(mapper)

	# vtkRenderer provides an abstract specification for renderers. A renderer is an 
	# object that controls the rendering process for objects. Rendering is the process 
	# of converting geometry, a specification for lights, and a camera view into an image.
	renderer = vtk.vtkRenderer()
	renderer.AddActor(actor)
	renderer.SetBackground(0.1, 0.2, 0.4)

	# create a window for renderers to draw into
	# vtkRenderWindow is an abstract object to specify the behavior of a rendering window. 
	# A rendering window is a window in a graphical user interface where renderers draw their images.
	render_window = vtk.vtkRenderWindow()
	render_window.AddRenderer(renderer)

	# Automatically set up the camera based on the visible actors.
	renderer.ResetCamera()

	


	interactive_window = vtk.vtkRenderWindowInteractor()
	interactive_window.SetRenderWindow(render_window)

	style = vtk.vtkInteractorStyleTrackballCamera()
	interactive_window.SetInteractorStyle(style)

	# # create the scalar_bar
	# scalar_bar = vtk.vtkScalarBarActor()
	# scalar_bar.SetOrientationToHorizontal()
	# scalar_bar.SetLookupTable(lut)

	# # create the scalar_bar_widget
	# scalar_bar_widget = vtk.vtkScalarBarWidget()
	# scalar_bar_widget.SetInteractor(interactive_window)
	# scalar_bar_widget.SetScalarBarActor(scalar_bar)
	# scalar_bar_widget.On()

	render_window.Render()

	picked_coords = []
	# global_p_coords = [0,0,0]
	picked_point = [0,0,0]

	def mark(x,y,z):
		"""mark the picked location with a sphere"""
		global picked_point
		# print(x,y,z)
		sphere = vtk.vtkSphereSource()
		sphere.SetRadius(0.02)
		res = 20
		sphere.SetThetaResolution(res)
		sphere.SetPhiResolution(res)
		sphere.SetCenter(x,y,z)
		sphere.Update()

		mapper_sphere = vtk.vtkPolyDataMapper()
		mapper_sphere.SetInputData(sphere.GetOutput())

		marker = vtk.vtkActor()
		marker.SetMapper(mapper_sphere)

		renderer.AddActor(marker)
		render_window.AddRenderer(renderer)

		marker.GetProperty().SetColor( (1,0,0) )
		interactive_window.Render()
		render_window.Render()

		picked_point = [x,y,z]
		print('test1: ', picked_point)
		
		return picked_point



	def pick_cell(renwinInteractor, event):
		global picked_point

		x, y = renwinInteractor.GetEventPosition()

		picker = vtk.vtkCellPicker()
		picker.SetTolerance(0.0005)
		picker.Pick(x, y, 0, renderer)
		vid =  picker.GetCellId()
		pid = picker.GetPointId()
		if vid!=-1:
		    sid = picker.GetSubId()
		    pcoords = picker.GetPCoords()
		    global_p_coords = data_CL.GetPoint(pid)
		    print(vid, sid, pid, pcoords, global_p_coords)
		    dataSet = picker.GetDataSet()
		    pnt = dataSet.GetPoint(vid)
		    picked_point = mark(*global_p_coords)
		    print('test2: ', picked_point)
		else:
		    print('swing and a miss')


	def keypress_callback(obj, ev):
		global picked_point
		key = obj.GetKeySym()
		if key == 'p':
			picked_coords.append(picked_point)
			print('Picked Point: ', picked_point)
			sphere = vtk.vtkSphereSource()
			sphere.SetRadius(0.025)
			res = 20
			sphere.SetThetaResolution(res)
			sphere.SetPhiResolution(res)
			sphere.SetCenter(picked_point[0],picked_point[1],picked_point[2])
			sphere.Update()

			mapper_sphere = vtk.vtkPolyDataMapper()
			mapper_sphere.SetInputData(sphere.GetOutput())

			marker = vtk.vtkActor()
			marker.SetMapper(mapper_sphere)

			renderer.AddActor(marker)
			render_window.AddRenderer(renderer)

			marker.GetProperty().SetColor( (0,1,0) )
			interactive_window.Render()
			render_window.Render()

		if key == 'Return':
			final_render_window = interactive_window.GetRenderWindow()
			final_render_window.Finalize()
			interactive_window.TerminateApp()
			del final_render_window, interactive_window


	interactive_window.AddObserver('LeftButtonPressEvent', pick_cell)

	interactive_window.AddObserver('KeyPressEvent', keypress_callback)

	interactive_window.Initialize()
	
	interactive_window.Start()

	print(picked_coords)
	return picked_coords, data_CL


def Find_CL_between_points(file_name, output_root):

	CL_main = 'vmtkcenterlines -ifile %s -ofile %scenterline_main.vtp' % (file_name, output_root)

	myPype = pypes.PypeRun(CL_main)

	print('Centerline successfully was created.\n Centerline File address: %scenterline_main.vtp' %output_root)

	output_centerline = '%scenterline_main.vtp' %output_root

	output_centerline_vtk = myPype.GetScriptObject('vmtkcenterlines','0').Centerlines

	print("Output centerline type: ", type(output_centerline_vtk))

	CL_points = VN.vtk_to_numpy(output_centerline_vtk.GetPoints().GetData())

	print(CL_points)

	return output_centerline, output_centerline_vtk, CL_points


def write_CL(path, CL):
    with open(path, "wt") as file:
        for row in CL:
            for item in row:
                file.write(str(item) + " ")
            file.write("\n")
	

def Get_WSS_LCS(path):

	print('Loading', path)

	lookup_point_str = 'POINTS'
	lookup_table_str = 'LOOKUP_TABLE'

	# l = [[float(num) for num in line.split(' ') if num != '\n'] for line in file_text[0:]]
	with open(root_dir+filename_WSS_LCS) as myFile:
		for num, line in enumerate(myFile, 1):
			if lookup_point_str in line:
				print('Coordinates found at line:', num)
				coords_line_id = num
			if lookup_table_str in line:
				print('Lookup table found at line:', num)
				lookup_table_line_id = num


	with open(root_dir+filename_WSS_LCS) as fo:
		file_text = fo.readlines()

	print(file_text[coords_line_id-1])
	num_points = [int(num) for num in file_text[coords_line_id-1].split(' ') if num.isdigit() and num != '\n']
	print(num_points[0])

	coords = np.asarray([[float(num) for num in line.split(' ') if num != '\n'] for line in file_text[coords_line_id:coords_line_id+num_points[0] ]])
	tags = np.asarray([[float(num) for num in line.split(' ') if num != '\n'] for line in file_text[lookup_table_line_id:lookup_table_line_id+num_points[0] ]])
	print(coords, len(coords))
	print(tags, len(tags))

