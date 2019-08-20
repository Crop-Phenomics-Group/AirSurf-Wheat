
from flask import Flask, render_template, url_for
from gevent.pywsgi import WSGIServer

class Interface(object):

    instance = None

    @staticmethod
    def getInstance():
        if Interface.instance is None:
            Interface.instance = Interface()
        return Interface.instance

    def __init__(self):
        super().__init__()
        self.app = Flask(__name__)
        self.port = 80
        self.host = '0.0.0.0'


        self.set_routes()

        # self.upload_folder = utils.make_dir('/tmp/phenobox/upload')
        # self.app.config['UPLOAD_FOLDER'] = self.upload_folder

    def run(self):
        http_server = WSGIServer(('', self.port), self.app)
        http_server.serve_forever()

    def makeHeader(self):

        return render_template('head.html', css=url_for('static', filename='style.css'),
                               logo=url_for('static', filename='logo.png'),
                               datatables_css=url_for('static', filename='datatables.min.css'),
                               datatables_js=url_for('static', filename='datatables.min.js'),

                               bootstrap_css=url_for('static', filename='bootstrap.min.css'))

    def makeFooter(self):
        return render_template('foot.html', jquery=url_for('static', filename='jquery.min.css'),
                               popper=url_for('static', filename='popper.min.js'),
                               bootstrap_js=url_for('static', filename='bootstrap.min.js'))

    def set_routes(self):

        @self.app.route("/", methods=['GET', 'POST'])
        def home():
            return self.makeHeader() + self.makeFooter()