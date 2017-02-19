var express = require('express'),
	stylus = require('stylus'),
	logger = require('morgan'),
	bodyParser = require('body-parser'),
	mongoose = require('mongoose'),
	PythonShell = require('python-shell'),
	path = require('path'),
	formidable = require('formidable'),
	fs = require('fs');

var env = process.env.NODE_ENV = process.env.NODE_ENV || 'development';

var app = express();

function compile(str, path) {
	return stylus(str).set('filename', path);
}

app.set('views', __dirname + '/server/views');
app.set('view engine', 'jade');
app.use(logger('dev'));
app.use(bodyParser.urlencoded({
	extended: true
}));
app.use(bodyParser.json());
app.use(stylus.middleware({
	src: __dirname + '/public',
	compile: compile
}));
app.use(express.static(__dirname + '/public'));


if (env === 'development') {
	mongoose.connect('mongodb://localhost/deepscan');
} else {
	mongoose.connect('mongodb://admin:password@ds153719.mlab.com:53719/deepscan');
}

var db = mongoose.connection;
db.on('error', console.error.bind(console, 'connection error...'));
db.once('open', function callback() {
	console.log('deepscan db opened');
	console.log('Environment: ' + env);
});
var messageSchema = mongoose.Schema({
	message: String
});
var Message = mongoose.model('Message', messageSchema);
var mongoMessage;
Message.find().exec(function (err, messageDoc) {
	mongoMessage = messageDoc[0].message;
});

app.get('/partials/:partialPath', function (req, res) {
	res.render('partials/' + req.params.partialPath);
});

app.get('/', function (req, res) {
	res.render('index', {
		mongoMessage: mongoMessage
	});
});

app.get('/api/endpoint', function (req, res) {
	var pyshell = new PythonShell('/python/test.py');
	pyshell.send("69");
	pyshell.on('message', function (message) {
		res.json({
			data: message
		});
	});
});

app.post('/api/upload', function (req, res) {

	// create an incoming form object
	var form = new formidable.IncomingForm();

	// specify that we want to allow the user to upload multiple files in a single request
	form.multiples = true;

	// store all uploads in the /uploads directory
	form.uploadDir = path.join(__dirname, '/python');

	// every time a file has been uploaded successfully,
	// rename it to it's orignal name
	form.on('file', function (field, file) {
		fs.rename(file.path, path.join(form.uploadDir, 'input.png'));
	});

	// log any errors that occur
	form.on('error', function (err) {
		console.log('An error has occured: \n' + err);
	});

	// once all the files have been uploaded, send a response to the client
	form.on('end', function () {
		res.end('success');
	});

	// parse the incoming request containing the form data
	form.parse(req);
});

app.get('/api/scantest', function (req, res) {
	var pyshell = new PythonShell('/python/deep_scan.py');
	pyshell.on('message', function (message) {
		//OUTPUT deep_scan.py
		console.log(message);
	});
});

var port = process.env.PORT || 3030;
app.listen(port);
console.log('Listening on port ' + port + '...');