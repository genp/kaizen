// JS for populating active querty form. 
// Fills in exemplar images, gallery of detections.
// Users can click on images to mark them as not similar to exemplars.

var exemplar_imgs = []
var exemplar_neg_imgs = []
var detected_imgs = []
var patch_size = 100.0
var row_width = 5;
hit_data["exemplar.size"] = [];
$("#exemplar-container").append('<div id="exemplars" class="patch-container" style="display:inline-block"></div>');
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

var gallery = $("#gallery-container");

var total = Math.min(10,hit_data.positives.length)+hit_data.queries.length; 

if ("negatives" in hit_data) {
    $("#exemplar-container").append('<div id="spacer"  style="width:100px; display:inline-block"></div>');
    $("#exemplar-container").append('<div id="exemplars-neg"  class="patch-container" style="display:inline-block"></div>');
    $("#exemplars-neg").empty().append("<h4><font color='990000'>These are incorrect selections: </font></h4>");
    total += Math.min(5,hit_data.negatives.length)
}

$(window).load(function() {    
    $("#keyword").html(keyword);
    $("#commit").prop("disabled",true);
    $("#radios").radiosToSlider();
    for (var index = 0; index < Math.min(10,hit_data.positives.length); index++) {
	exemplar_imgs[index] = new Image();
	exemplar_imgs[index].src = patch_base + hit_data.positives[index];
	$(exemplar_imgs[index]).load({ind: index}, function(event) { 
	    load_exemplar_img(event.data.ind); 
	});    	
    }
    if ("negatives" in hit_data) {
	for (var index = 0; index < Math.min(10,hit_data.negatives.length); index++) {
	    exemplar_neg_imgs[index] = new Image();
	    exemplar_neg_imgs[index].src = patch_base + hit_data.negatives[index];
	    $(exemplar_neg_imgs[index]).load({ind: index}, function(event) { 
		load_exemplar_neg_img(event.data.ind); 
	    });
	}
    }
    // loadPage();
    for (var index = 0; index < hit_data.queries.length; index++) {
    	detected_imgs[index] = new Image();
    	detected_imgs[index].src = patch_base + hit_data.queries[index];
    	$(detected_imgs[index]).load({ind: index}, function(event) { 
    	    load_gallery_img(event.data.ind); 
    	});
    }
});

function load_when_ready() {
    total--;
    if (total == 0)
	loadPage();
}

function load_exemplar_img(index) {
    img_name = patch_base + hit_data.positives[index];
    $('#exemplars').append('<img class="patch" src="'+img_name+'"/>');
    load_when_ready();
}

function load_exemplar_neg_img(index) {
    img_name = patch_base + hit_data.negatives[index];
    $('#exemplars-neg').append('<img class="patch" src="'+img_name+'"/>');
    load_when_ready();
}

function load_gallery_img(index) {
    i = Math.floor(index/row_width);
    j = index % row_width;

    img_name = patch_base + hit_data.queries[i*row_width+j];
    $('#gallery-' + i +'-'+j).html('<img class="patch" src="'+img_name+'"/>');
    load_when_ready();
}


function loadPage() {
    start = new Date;
    get_location();

    /* Given mturk details, put into form. */
    var queryParameters = get_url_vars();
    assignmentId = queryParameters.assignmentId;
    workerId = queryParameters.workerId;
    hitId = queryParameters.hitId;
    if (assignmentId)
      $("#mturk").val(assignmentId+","+workerId+","+hitId)

    $(".image-patch").draggable({
        revert: "invalid",
        containment: "document",
        helper: "original",
        distance: 5,
        opacity: .9
    });

    $("#cluster").droppable({
        accept: "#gallery .image-patch",
        activeClass: "ui-state-highlight",
        drop: function( event, ui ) {
            addToCluster( ui.draggable );
        }
    });

    $('.image-patch').click(function() {
        addToCluster(this);
    });

    $("#clear").click(function(ev) {
	ev.preventDefault();
        $("#result").empty();
        $("#results").val('');
        $("#cluster .image-patch").each(function() {
            removeFromCluster(this);
        });
    });

    $("#cluster-background").css("height", $("#gallery-container").css("height"));
    $("#cluster-background").css("width", $("#gallery-container").css("width"));

    $("#commit").prop("disabled",false);

};

function addToCluster(obj) {
    var coords = $(obj).attr('id').split('-');
    var i = parseInt(coords[1]), j = parseInt(coords[2]);

    $('#gallery-'+i+'-'+j).css('visibility','hidden');
    cluster_patches.push([i,j]);
    reflow();
}

function removeFromCluster(obj) {
    var coords = $(obj).attr('id').split('-');
    var i = parseInt(coords[1]), j = parseInt(coords[2]);

    $('#gallery-'+i+'-'+j)
      .css('visibility','visible')
      .css('top', 'auto')
      .css('left', 'auto');
    for (var index = 0; index < cluster_patches.length; index++) {
        if (cluster_patches[index][0] == i && cluster_patches[index][1] == j) {
            cluster_patches.splice(index, 1);
        }
    }
    reflow();
}

function reflow() {
    var cluster = $('#cluster-container');
    cluster.empty();

    var cluster_rows = cluster_patches.length / row_width;
    var index = 0;
    for (var i = 0; i < cluster_rows; i++) {
        cluster.append('<ul id="clusterrow' + i + '" class="patch-row"> </ul>');
	var cluster_row = $('#clusterrow' + i);
        for (var j = 0; j < row_width; j++) {

            if (index < cluster_patches.length) {

                var pi = cluster_patches[index][0], pj = cluster_patches[index][1];
		img_name = patch_base + hit_data.queries[pi*row_width+pj];

		cluster_row.append('<li id="cluster-' + pi + '-'+ pj + '" class="image-patch"> <img class="patch" src="' + img_name + '"\></li>');
            }
            index++;
        }
    }

    $( "#cluster .image-patch").draggable({
        revert: "invalid",
        containment: "document",
        helper: "original",
        distance: 5,
        opacity: .9
    });

    $("#gallery").droppable({
        accept: "#cluster .image-patch",
        activeClass: "ui-state-highlight",
        drop: function( event, ui ) {
            removeFromCluster( ui.draggable );
        }
    });

    $('#cluster .image-patch').click(function() {
        removeFromCluster(this);
    });

    // This is if the user has to click minimum of patches
    if (cluster_patches.length >= 0) {
        $("#commit").removeAttr('disabled');
        $("#commit-warning").html("");
    } else {
        $("#commit").attr('disabled', 'disabled');
        $("#commit-warning").html("Please click at least " + 
		(5 - (cluster_patches.length ? cluster_patches.length : 0)) + " more images.");
    }
}

$("#commit").click(function (ev) {
    /* Prevent direct form submission */
    ev.preventDefault();

    $('#title').html("<h1> Form submiting..... </h1>");
    $(this).prop("disabled",true);

    /* Aggregate results */
    
    var selected_img_name = [];
    for (var index = 0; index < cluster_patches.length; index++) {
        var pi = cluster_patches[index][0], pj = cluster_patches[index][1];
    	    selected_img_name.push(hit_data.queries[pi*row_width+pj]);
    }

    /* This checks if any of the gallery images are in cluster_patches, if not adds them to results */
    var results_img_name = [];
    for (var index = 0; index < hit_data.queries.length; index++) {
         var pi = Math.floor(index/row_width), pj = index % row_width;
         
         indexes = $.map(cluster_patches, function(obj, index) {
                         if(obj[0] == pi && obj[1] == pj) {
                           return index;
                         }
         })
         
         if(indexes.length > 0) continue;
         
         results_img_name.push(hit_data.queries[pi*row_width+pj]);
    }

    /** Stash the results in the form */
    $("#neg_patches").val(results_img_name);
    $("#pos_patches").val(selected_img_name);
    $("#confidence").val($('input[name="options"]:checked').val());
    $("#time").val(time = (new Date - start) / 1000);
    $("#location").val(ip);
    $("#nationality").val(nationality);

    var data = $("#hit_response").serialize();
    var that = this;
    $.ajax({
    	type: "POST",
    	url: "/classifier/update/"+$("#classifier").val(),
    	data: data,
    	success: function() {
    	    //if workerId, submit to mturk
    	    //else redirect to previous page

	    if (user.lastIndexOf('mturk_', 0) === 0) {
	    	// console.dir($(that).parent().serialize());
	        // console.dir($("#mturk").serialize());
	    	$(that).parent().submit();
	    } else {	    
    	    	history.go(-1);
    	    	// if the history redirect doesn't work
    	    	//window.location.href = "/classifier/";
	    }
    	}
    });
});



/**
 * Return JavaScript object of query parameters
*/ 
function get_url_vars() {
    var vars = [], hash;
    var hashes = window.location.href.
                 slice(window.location.href.indexOf('?') + 1).split('&');

    for (var i = 0; i < hashes.length; i++) {
        hash = hashes[i].split('=');
        vars[hash[0]] = hash[1];
    }
    return vars;
}

function get_location() {
    $.ajax( { 
	url: '//freegeoip.net/json/', type: 'POST', 
	dataType: 'jsonp',
	success: function(location) {
	    // city = location.city;
	    nationality = location.country_name;
	    // longitude = location.longitude;
	    // longitude = location.latitude;
	    ip = location.ip;
	}
    } );
}


