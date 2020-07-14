from flask import request, Flask
from scipy.stats import ttest_1samp
import numpy as np

app = Flask(__name__)


@app.route('/t_test', methods=['POST'])
def model_train():
    print("Null Hypothesis:μ=μ0=30，α=0.05")
    str = request.form.get("str")
    srt_list = str.split(',')
    list1 = list(map(float, srt_list))

    global_mean = float(request.form.get("global_mean"))
    # ages = [25, 36, 15, 40, 28, 31, 32, 30, 29, 28, 27, 33, 35]

    t = (np.mean(list1) - global_mean) / (np.std(list1, ddof=1) / np.sqrt(len(list1)))

    ttest, pval = ttest_1samp(list1, global_mean)
    print(t, ttest)
    if pval < 0.05:
        return "Reject the Null Hypothesis."
    else:
        return "Accept the Null Hypothesis."


if __name__ == '__main__':
    app.run(debug=True, port=9021)
