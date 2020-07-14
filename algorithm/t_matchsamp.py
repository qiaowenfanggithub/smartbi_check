from scipy.stats import ttest_rel
from flask import request, Flask


#s1 = [620.16,866.50,641.22,812.91,738.96,899.38,760.78,694.95,749.92,793.94]
#s2 = [958.47,838.42,788.90,815.20,783.17,910.92,758.49,870.80,826.26,805.48]

app = Flask(__name__)


@app.route('/t_test', methods=['POST'])
def model_train():
	str1 = request.form.get("str1")
	srt_list1 = str1.split(',')
	list1 = list(map(float, srt_list1))

	str2 = request.form.get("str2")
	srt_list2 = str2.split(',')
	list2 = list(map(float, srt_list2))
	print("Null Hypothesis:mean(s1)=mean(s2)，α=0.05")
	ttest, pval = ttest_rel(list1, list2)
	if pval < 0.05:
		return "Reject the Null Hypothesis."
	else:
		return "Accept the Null Hypothesis."


if __name__ == '__main__':
    app.run(debug=True, port=9022)