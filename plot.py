from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

trig_data = {0.5: {'rec': 0.5472, 'prec': 0.1268, 'F1': 0.2058},
 0.6: {'rec': 0.5462, 'prec': 0.1256, 'F1': 0.2042},
 0.7: {'rec': 0.5474, 'prec': 0.1255, 'F1': 0.2041},
 0.8: {'rec': 0.5525, 'prec': 0.1252, 'F1': 0.204},
 0.9: {'rec': 0.5634, 'prec': 0.1199, 'F1': 0.1976},
 0.95: {'rec': 0.5804, 'prec': 0.1125, 'F1': 0.1883}}

arg_data = {0.5: {'rec': 0.7057, 'prec': 0.0514, 'F1': 0.0958},
 0.6: {'rec': 0.7052, 'prec': 0.0514, 'F1': 0.0958},
 0.7: {'rec': 0.7041, 'prec': 0.0512, 'F1': 0.0955},
 0.8: {'rec': 0.7025, 'prec': 0.0514, 'F1': 0.0957},
 0.9: {'rec': 0.7044, 'prec': 0.0514, 'F1': 0.0958},
 0.95: {'rec': 0.7054, 'prec': 0.0514, 'F1': 0.0957}}

r = trig_data.keys()
r.sort()

trig_rec = [trig_data[i]['rec'] for i in r]
trig_prec = [trig_data[i]['prec'] for i in r]
trig_F1 = [trig_data[i]['F1'] for i in r]

arg_rec = [arg_data[i]['rec'] for i in r]
arg_prec = [arg_data[i]['prec'] for i in r]
arg_F1 = [arg_data[i]['F1'] for i in r]

def main():
	# fig, ax = plt.subplots()
	# ax.plot(r, trig_rec, 'r--o', label='Recall')
	# ax.plot(r, trig_prec, 'b--s', label='Precision')
	# ax.plot(r, trig_F1, 'g--^', label='F1')
	# plt.xlabel('subsampling rate (%)', fontsize=15)
	# plt.ylabel('score (%)', fontsize=15)
	# plt.title('Evaluation of Naive Bayes for trigger prediction')
	# plt.ylim(0,1)
	# plt.grid(True)
	# ax.legend(loc='upper right')
	# plt.savefig("trig_eval_nb.png")
	# plt.show()

	fig, ax = plt.subplots()
	ax.plot(r, arg_rec, 'r--o', label='Recall')
	ax.plot(r, arg_prec, 'b--s', label='Precision')
	ax.plot(r, arg_F1, 'g--^', label='F1')
	plt.xlabel('subsampling rate (%)', fontsize=15)
	plt.ylabel('score (%)', fontsize=15)
	plt.title('Evaluation of Naive Bayes for argument prediction')
	plt.ylim(0,1)
	plt.grid(True)
	ax.legend(loc='upper right')
	plt.savefig("arg_eval_nb.png")
	plt.show()

if __name__ == '__main__':
	main()