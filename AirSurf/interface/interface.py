
from flask import Flask, render_template, url_for, request
from gevent.pywsgi import WSGIServer
import os
from threading import Thread

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

        BrowserLauncher.getInstance().start()

        http_server = WSGIServer(('', self.port), self.app)
        http_server.serve_forever()

    def makeHeader(self):

        return render_template('head.html', css=url_for('static', filename='style.css'),
                               logo=url_for('static', filename='logo.png'),
                               datatables_css=url_for('static', filename='datatables.min.css'),
                               datatables_js=url_for('static', filename='datatables.min.js'),

                               bootstrap_css=url_for('static', filename='bootstrap.min.css'))

    def makeFooter(self):
        return render_template('foot.html', jquery=url_for('static', filename='jquery.min.js'),
                               popper=url_for('static', filename='popper.min.js'),
                               bootstrap_js=url_for('static', filename='bootstrap.min.js'))

    def set_routes(self):

        @self.app.route("/", methods=['GET', 'POST'])
        def home():
            return self.makeHeader() + render_template('start.html') + self.makeFooter()

        @self.app.route("/run", methods=['GET', 'POST'])
        def run():
            print(request.form)
            return self.makeHeader() + render_template('processing.html') + self.makeFooter()

import time

class BrowserLauncher(Thread):

    instance = None

    @staticmethod
    def getInstance():
        if BrowserLauncher.instance is None:
            BrowserLauncher.instance = BrowserLauncher()
        return BrowserLauncher.instance

    def __init__(self):
        super().__init__()

    def run(self):
        time.sleep(2)
        os.system('explorer "http://localhost"')