// JS for populating active questy form. 
// Fills in exemplar images, gallery of detections.
// Users can click on images to mark them as not similar to exemplars.

var exemplar_imgs = []
var detected_imgs = []
var patch_size = 100.0
var row_width = 5;
examples["exemplar.size"] = [];
$("#exemplars").empty().append("<h3>Examples: </h3>");
var num_patches = detected_imgs.length;
var num_rows = num_patches / row_width;
var cluster_patches = [];
var start = 0;
var assignmentId = '';
var workerId = '';
var hitId = '';
var user = '';
var confidence = 0;
var ip = 0;
var nationality = '';

$("#cluster").
  css('min-height', 90 * num_rows + 45).css('min-width', 90 * row_width + 30);
$("#gallery").
  css('min-height', 90 * num_rows + 45).css('min-width', 90 * row_width + 30);

$(window).load(function() {    
    $("#commit").prop("disabled",true);
    $("#radios").radiosToSlider();
    for (var index = 0; index < Math.min(10,examples["exemplar.file"].length); index++) {
	exemplar_imgs[index] = new Image();
	exemplar_imgs[index].src = img_baseurl + examples["exemplar.file"][index];
	$(exemplar_imgs[index]).load({ind: index}, function(event) { 
	    load_exemplar_img(event.data.ind); 
	});    	
    }

    for (var index = 0; index < neg["TopConf.file"].length; index++) {
    	detected_imgs[index] = new Image();
    	detected_imgs[index].src = img_baseurl + neg["TopConf.file"][index];
    	$(detected_imgs[index]).load({ind: index}, function(event) { 
    	    load_gallery_img(event.data.ind); 
    	});

    }

    for (var index = 0; index < pos["TopConf.file"].length; index++) {
    	detected_imgs[index] = new Image();
    	detected_imgs[index].src = img_baseurl + pos["TopConf.file"][index];
    	$(detected_imgs[index]).load({ind: index}, function(event) { 
    	    load_cluster_img(event.data.ind); 
    	});

    }


});

function load_exemplar_img(index) {


    img_name = img_baseurl + examples["exemplar.file"][index];
    x = examples["exemplar.loc"][index][0];
    y = examples["exemplar.loc"][index][1];
    width = examples["exemplar.loc"][index][2];

    img_width = exemplar_imgs[index].width;

    $('#exemplars').append('<div style="background-image: url(' + img_name + '); width:100px; height:100px; background-size:'+100.0*img_width/width+'px; background-position:-'+100.0*y/width+'px -'+100.0*x/width+'px; float: left; border: 5px solid #222;">&nbsp;</div>');


}
function load_gallery_img(index) {

    i = Math.floor(index/row_width);
    j = index % row_width;

    img_name = img_baseurl + neg["TopConf.file"][i*row_width+j];
    x = neg["TopConf.loc"][i*row_width+j][0];
    y = neg["TopConf.loc"][i*row_width+j][1];
    width = neg["TopConf.loc"][i*row_width+j][2];
    img_width = detected_imgs[index].width;

    $('#gallery-' + i +'-'+j).html('<div style="background-image: url(' + img_name + '); width:100px; height:100px; background-size:'+100.0*img_width/width+'px; background-position:-'+100.0*y/width+'px -'+100.0*x/width+'px; float: left;">&nbsp;</div>');


}
function load_cluster_img(index) {

    i = Math.floor(index/row_width);
    j = index % row_width;

    img_name = img_baseurl + pos["TopConf.file"][i*row_width+j];
    x = pos["TopConf.loc"][i*row_width+j][0];
    y = pos["TopConf.loc"][i*row_width+j][1];
    width = pos["TopConf.loc"][i*row_width+j][2];
    img_width = detected_imgs[index].width;

    $('#cluster-' + i +'-'+j).html('<div style="background-image: url(' + img_name + '); width:100px; height:100px; background-size:'+100.0*img_width/width+'px; background-position:-'+100.0*y/width+'px -'+100.0*x/width+'px; float: left;">&nbsp;</div>');


}


