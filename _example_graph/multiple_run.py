import os

for num in range(0, 10):
	cmd = 'python graph_drawing.py -i test_result' + str(num) + '.graph -o temp' + str(num) + '.png'
	print(cmd)
	os.system(cmd)

