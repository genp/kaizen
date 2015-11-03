// JS for operating and submitting active query form. 
// Users can click on images to mark them as not similar to exemplars.

var start = 0;
var assignmentId = '';
var workerId = '';
var hitId = '';
var user = '';
var ip = 0;
var nationality = '';

// Turn off the patch's visibility, turn on its opposite
function mark_patch(obj, type) {
  $(obj).css('visibility', 'hidden');
  id = $(obj)[0].id
  pid = id.split("-")[2]
  $('#'+type+'-patch-'+pid).css('visibility', 'visible');
}


function active_query() {    
    start = new Date;
    $("#radios").radiosToSlider();
    get_location();

    /* Given mturk details, put into form. */
    var queryParameters = get_url_vars();
    assignmentId = queryParameters.assignmentId;
    workerId = queryParameters.workerId;
    hitId = queryParameters.hitId;
    if (assignmentId)
      $("#mturk").val(assignmentId+","+workerId+","+hitId)

    $('#gallery .patch').click(function() {
        mark_patch(this, "neg");
    });

    $('#cluster .patch').click(function() {
        mark_patch(this, "pos");
    });

    $("#clear").click(function(ev) {
        ev.preventDefault();
        clear();
        $("#cluster .image-patch").each(function() {
            removeFromCluster(this);
        });
    });

    $("#commit").click(function (ev) {
        /* Prevent direct form submission */
        ev.preventDefault();
        /* Prevent multiple clicks */
        $(this).prop("disabled", true);

        pos_patches = [];
        neg_patches = [];
        $('#gallery .patch').each(function(i,img) {
          visible = $(img).css('visibility') == 'visible'
          pid = img.id.split("-")[2]
          if (visible) {
            pos_patches.push(pid)
          } else {
            neg_patches.push(pid)
          }
        });
        
        /** Stash the results in the form */
        $("#time").val(time = (new Date - start) / 1000);
        $("#location").val(ip);
        $("#nationality").val(nationality);
        $("#confidence").val($('input[name="options"]:checked').val());
        $("#pos_patches").val(pos_patches);
        $("#neg_patches").val(neg_patches);
        
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
}


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


