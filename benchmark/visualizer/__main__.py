from benchmark.visualizer.app import create_app

app = create_app()
app.run(host='0.0.0.0', debug=True, port=8052)