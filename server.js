var express = require('express'),
	stylus = require('stylus'),
	logger = require('morgan'),
	bodyParser = require('body-parser'),
	mongoose = require('mongoose');

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
	res.json({
		data: "this is a test api endpoint"
	});
});

app.post('/api/scan', function (req, res) {

});

var port = process.env.PORT || 3030;
app.listen(port);
console.log('Listening on port ' + port + '...');