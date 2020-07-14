# 卡方检验
from flask import request, Flask
from scipy.stats import chi2_contingency
import numpy as np

app = Flask(__name__)


@app.route('/t_test', methods=['POST'])
def model_train():
    file_predict = request.files['kfdata']
    # 读取csv文件
    predict_data = np.loadtxt(file_predict, delimiter=",")
    kf_data = predict_data[:, :]
    print("----------", kf_data)
    # kf_data = np.array([[11.7, 8.7, 15.4, 8.4], [18.1, 11.7, 24.3, 13.6],
    #                    [26.9, 20.3, 37, 19.3], [41, 30.9, 54.6, 35.1],
    #                   [66, 54.3, 71.1, 50]])
    V, pval, dof, expected = chi2_contingency(kf_data)

    if pval < 0.05:
        #return ("卡方值为%，小于0.05", pval)
        return str(pval)
    else:
        #return ("卡方值为%，大于0.05", pval)
        return str(pval)


if __name__ == '__main__':
    app.run(debug=True, port=9023)
