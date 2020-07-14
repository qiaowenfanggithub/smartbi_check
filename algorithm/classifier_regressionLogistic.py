from flask import request, Flask, jsonify
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import numpy as np

app = Flask(__name__)

ALLOWED_EXTENSIONS = set(['txt', 'text', 'excel'])


@app.route('/train', methods=['GET', 'POST'])
def model_train():
    if request.method == 'POST':
        # 使用request上传文件，其中‘file'表示前端表单的Key值；也可以使用request.files['file']
        f = request.files['file']
        # 判断是否上传成功
        if f is None:
            return jsonify({"Status": "Error 0000", "Msg": "没有上传文件，请重新上传!"})
        # 检查文件后缀名是否是文本文件
        if not allow_file(f.filename):
            return jsonify({"Status": "Error 9999", "Msg": "文件格式不支持，仅支持如下图片格式:'txt', 'text', 'excel'。"})
        # 使用plt将图片读入为数组
        data = np.loadtxt(f, delimiter=',')

        len = data.shape[1] - 1
        data_X = data[:, 0:len]
        data_y = data[:, len]
        # 数据分割
        x_train, x_test, y_train, y_test = train_test_split(data_X, data_y, random_state=666)

        clf = LogisticRegression()
        clf.fit(x_train, y_train)

        # 模型评估
        print(clf.score(x_train, y_train))
        print(clf.score(x_test, y_test))

        # 模型的保存与持久化
        joblib.dump(clf, "logistic_lr.model")  # 将训练后的线性模型保存

        return '模型训练完成！'


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    file_model = request.files['model']

    file_predict = request.files['predict']
    # 读取csv文件
    predict_data = np.loadtxt(file_predict, delimiter=",")
    predict_data1 = predict_data[:, :]

    print(predict_data)

    model = joblib.load(file_model)
    result = model.predict(predict_data1)

    return " ".join('%s' % id for id in result)


# 检查文件后缀名是否是文本文件
def allow_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == '__main__':
    app.run(debug=True, port=9000)
