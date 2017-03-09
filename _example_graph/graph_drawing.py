import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys, getopt

def main(argv):

	inputfile = ''
	outputfile = ''

	try:
		opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
	except getopt.GetoptError:
		print('test.py -i <inputfile> -o <outputfile>')
		sys.exit(2)

	for opt, arg in opts:
		if opt == '-h':
			print('graph_drawing.py -i <inputfile> -o <outputfile>')
			print(arg)
			sys.exit()
		elif opt in ("-i", "--ifile"):
			inputfile = arg
			#print(arg)
		elif opt in ("-o", "--ofile"):
			outputfile = arg
			#print(arg)

	print('Input file is ', inputfile)
	print('Output file is ', outputfile)

	file_object = open(inputfile, 'r')

	graph_name = file_object.readline()
	x_label = file_object.readline()
	num_x = file_object.readline()
	x_data = file_object.readline()
	y_label = file_object.readline()
	num_y = file_object.readline()
	y_data = file_object.readline()

	#print(x_data )
	#print(y_data )

	x_temp = [float(x) for x in x_data.split(' ')]
	y_temp = [float(y) for y in y_data.split(' ')]

	y = y_temp 
	x = x_temp 

	fig = plt.figure()
	ax = plt.subplot(111)
	ax.plot(x, y)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.title(graph_name)
	ax.legend()

	#plt.show()
 
	fig.savefig(outputfile)

if __name__ == "__main__":
	main(sys.argv[1:])

'''

python graph_drawing.py -i test_graph.graph -o test_graph.png

test_graph.graph

Graph Name
X-axis Label
5
1 2 3 4 5
Y-axis Label
5
1 2 3 2 2

'''
