# import pymysql
# pymysql.install_as_MySQLdb()
#
# # import blueprint
# from flask import Flask
# from flask import Blueprint as blueprint
# from flasgger import Swagger
#
# import config
#
# app = Flask(__name__)
# app.config.from_object(config)
#
# swagger_config = Swagger.DEFAULT_CONFIG
# swagger_config['title'] = config.SWAGGER_TITLE    # 配置大标题
# swagger_config['description'] = config.SWAGGER_DESC    # 配置公共描述内容
# swagger_config['host'] = config.SWAGGER_HOST    # 请求域名
#
# # swagger_config['swagger_ui_bundle_js'] = '//unpkg.com/swagger-ui-dist@3/swagger-ui-bundle.js'
# # swagger_config['swagger_ui_standalone_preset_js'] = '//unpkg.com/swagger-ui-dist@3/swagger-ui-standalone-preset.js'
# # swagger_config['jquery_js'] = '//unpkg.com/jquery@2.2.4/dist/jquery.min.js'
# # swagger_config['swagger_ui_css'] = '//unpkg.com/swagger-ui-dist@3/swagger-ui.css'
# Swagger(app, config=swagger_config)
#
# @blueprint.route('/register/', methods=['POST'])
# def register():
#     """
#     用户注册
#     ---
#     tags:
#       - 用户相关接口
#     description:
#         用户注册接口，json格式
#     parameters:
#       - name: body
#         in: body
#         required: true
#         schema:
#           id: 用户注册
#           required:
#             - username
#             - password
#             - inn_name
#           properties:
#             username:
#               type: string
#               description: 用户名.
#             password:
#               type: string
#               description: 密码.
#             inn_name:
#               type: string
#               description: 客栈名称.
#             phone:
#               type: string
#               description: 手机号.
#             wx:
#               type: string
#               description: 微信.
#
#     responses:
#       201:
#           description: 注册成功
#
#
#           example: {'code':1,'message':注册成功}
#       406:
#         description: 注册有误，参数有误等
#
#     """
#     pass