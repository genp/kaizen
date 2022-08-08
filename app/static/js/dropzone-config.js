Dropzone.options.datasetDrop = {
  maxFiles: 1,
  addRemoveLinks: true, 
  init: function() {
    this.on("success", function(file, response) {
        console.log(response);
        if (response.errors != null){
            console.info(response)
          alert(response.errors)
        }
        else
          window.location = response.url
    });
  }
}

var user_patches = {};
var img_ids = {}
var image_ratio_width = {}
var img_infos = {}

Dropzone.options.seedDrop = {
  addRemoveLinks: true,
  init: function() {
    this.on("success", function(file, response) {
        if (response.results == 0) {
            alert(response.errors);
            return;
        }
        img_ids[file.name] = response.results;
        readURL(response.results, true);
    });
    this.on("removedfile", function(file) {  
        removeBbox(img_ids[file.name]);
    });
  }
};

Dropzone.options.seedDropNeg = {
  addRemoveLinks: true,
  init: function() {
    this.on("success", function(file, response) {
        if (response.results == 0) {
            alert(response.errors);
            return;
        }
        console.info(`response: ${response}`);
        img_ids[file.name] = response.results;
        readURL(response.results, false);
    });
    this.on("removedfile", function(file) {  
        removeBbox(img_ids[file.name]);
    });
  }
};


//The pos_or_neg field is a boolean stating if the seedDropZone used was the one
//for positive or negative examples
function readURL(fileNumber, pos_or_neg) {
    addNewBboxSelector(fileNumber, pos_or_neg);
    displayImg(fileNumber, pos_or_neg);
}

function displayImg(fileNumber, pos_or_neg) {
    $("#" + fileNumber).on("load", function(){
	console.log("#"+fileNumber+' on load function');
        // get the int value number of px. This is assuming it is in px
        container_width = parseFloat($("#container").css("width").split("px")[0]);
        image_width = this.width;
        img_infos[fileNumber] = [this.width, this.height, pos_or_neg]

        if (image_width > container_width){
            ratio = container_width / image_width;
            $("#" + this.id + "_div").css("width", container_width + "px");
            $("#" + this.id).css("width", "100%");
        } else {
            ratio = 1.0;
        }
        image_ratio_width[fileNumber] = [ratio,image_width];

    });
    prefix = window.location.href.replace("keyword/new", "")
    $("#"+fileNumber).attr('src', prefix+'/blob/'+fileNumber);
}

function addNewBboxSelector(fileNumber, pos_or_neg) {
    createBbox(fileNumber, "img-container");
    addBboxHandles(fileNumber, pos_or_neg);
}

function createBbox(imgName, containerName) {
    $("#"+containerName).append($('<div>', { id: imgName+"_div"}));
    $("#"+imgName+"_div").html($('<img>', { id: imgName}));
}

function removeBbox(fileName) {
    $("#"+fileName).imgAreaSelect({remove:true});
    $("#"+fileName).hide();
    delete user_patches[fileName];
    console.dir(JSON.stringify(user_patches));
    removeSeedPreviews(fileName);
}

function addBboxHandles(fileName, pos_or_neg) {
    $("#"+fileName).imgAreaSelect({
        handles: true,
        aspectRatio: "1:1",
        parent: "#"+fileName+"_div",
        onSelectEnd: function (img, selection) {
            ratio = image_ratio_width[fileName][0];
                if (typeof user_patches[fileName] === 'undefined') {
                    user_patches[fileName] = [[Math.floor(selection.x1 / ratio), Math.floor(selection.y1 / ratio),Math.floor(selection.width / ratio), pos_or_neg]];
                }
                else{
                    user_patches[fileName].push([Math.floor(selection.x1 / ratio), Math.floor(selection.y1 / ratio), Math.floor(selection.width / ratio), pos_or_neg]);
                }   
            createSeedPreview(fileName, pos_or_neg);
        }
    });
}

function createSeedPreview(fileName, pos_or_neg) {
    count = user_patches[fileName].length-1;
    r = Math.floor((Math.random() * 100) + 1);

    //$("#keyword-container").append($('<div>', { id: fileName+"_"+count+"_"+r, class: "bbox"}));   
    s = (pos_or_neg == true ? "#positiveSeedsPreview" : "#negativeSeedsPreview"); 
    $(s).show()
    $(s).append($('<div>', { id: fileName+"_"+count+"_"+r, class: "bbox"}));
    img_width = image_ratio_width[fileName][1]
    console.log("img_width is :" + img_width);
    x = user_patches[fileName][count][0];
    y = user_patches[fileName][count][1];
    size = user_patches[fileName][count][2];
    
    $("#"+fileName+"_"+count+"_"+r).click( function () {
            removeSeedPreview(this);
        });
    
    $("#"+fileName+"_"+count+"_"+r).css({ "background-image": "url(" + $("#"+fileName).attr("src") + ")",  "width" : "100px", "height" : "100px",  "background-size": 100.0*img_width/size+"px", "background-position":"-"+100.0*x/size+"px -"+100.0*y/size+"px",  "border": "2px solid #222"});

    $("#"+fileName+"_"+count+"_"+r).hover( function () {
            $(this).css({ "border": "2px solid #C00"});
        },
        function () {
            $(this).css({ "border": "2px solid #222"});
        });

    console.dir(JSON.stringify(user_patches));

}

function removeSeedPreview(obj) {
    $(obj).hide();
    console.dir(obj.id.split("_"));
    info = obj.id.split("_");
    filename = info[0];
    count = parseInt(info[1]);
    if (user_patches[filename].length == 1) {
	delete user_patches[filename];
    } else {
	user_patches[filename].splice(count, 1);
    }
    console.dir(JSON.stringify(user_patches));
}

function removeSeedPreviews(fileName) {
    $.each( $("[id^='"+fileName+"_']"), function () {
            console.log(this.class);
            $(this).hide();
        });
}

var detect_blobs = [];
Dropzone.options.detectDrop = {
  addRemoveLinks: true,
  init: function() {
    this.on("success", function(file, response) {
        if (response.results == 0) {
            alert(response.errors);
            return;
        }
      detect_blobs.push(response.results)
      $("#blobs").val(detect_blobs.join(","));
    });
    this.on("removedfile", function(file) {  
        removeBbox(img_ids[file.name]);
    });
  }
};