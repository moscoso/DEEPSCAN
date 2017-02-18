angular.module('app', ['ngResource', 'ngRoute']);

angular.module('app').config(function ($routeProvider, $locationProvider) {
	$locationProvider.html5Mode(true);
	$routeProvider
		.when('/', {
			templateUrl: '/partials/main',
			controller: 'mainCtrl'
		});
	/*.otherwise({
				redirectTo: '/'
			});*/
});

angular.module('app').controller('mainCtrl', function ($scope, $http) {
	$scope.title = "DeepScan";
	$scope.description = "Make testing free with this futuristic Scantron Web App";
	$scope.testAPIdummy = "Getting /api/hello";
	$http.get("/api/endpoint")
		.then(function (response) {
			$scope.testResponse = response.data;
		});
});