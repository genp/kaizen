var user_patches = {};
var img_ids = {}
var image_ratio_width = {}



Dropzone.options.datasetDrop = {
  maxFiles: 1,
  addRemoveLinks: true, 
  init: function() {
    this.on("success", function(file, response) {
        console.log(response);
        window.location = response.url
    });
  }
}


/*
function that analyzes the image uploaded on the test page. Response contains a path to the image in the database.
*/
function analyzeImage(file) {

    // create random name for the image
    imgName = "bbox-"+Math.floor(Math.random()*1000).toString();

     // create the div to hold everything
    $("#img-container").append($('<div>', { id: imgName+"_div", class: "bbox"}));
    $("#img-container").append($('<h1> Select Seed Patch(es) </h1>'));
    // make the actual image element
    $("#"+imgName+"_div").html($('<img>', { id: imgName, usemap: "#"+imgName+"_map" }));
    // create the map for the image, used later to highlight patches
    $("#"+imgName+"_div").append($('<map>', {id: imgName+"_map", name: imgName + "_map"}));
    // load in the image to the img element
    $("#"+imgName).attr('src', '/blob/'+file);


    // initially, set the highlighting to be off. Will be turned on later
    $('#'+imgName).maphilight({ neverOn : true});
    

    // for now, there are just some hardcoded bounding boxes to be highlighted
    coordinates = [30,30,150,150];
    testImage(imgName, coordinates, 1.0, "horn");
    coordinates = [50,50,200,200];
    testImage(imgName, coordinates, 0.8, "horn");
    createScrollover("horn", imgName);

    coordinates = [80,0,100,40];
    testImage(imgName, coordinates, 1.0, "leaf");
    createScrollover("leaf", imgName);

    coordinates = [60,60,120,120];
    testImage(imgName, coordinates, -1.0, "beak");
    createScrollover("beak", imgName);
}


/*
coordinates is a size 4 array. 0 and 1 are top left of bounding box. 2 and 3 are bottom right.

imgName is the id of the image we're working with. ex: bbox-4352

confidence is a float, usually between -1 and 1, with higher values meaning there's a more confident detection.

for now, keyword is a string, the name of the keyword we're trying to recognize, but later it will probably involve 
hey keyword's ID, or other parameters as well, depending on how unique we need to make it
*/
function testImage(imgName, coordinates, confidence, keyword) {

    // set an intial border color. it changes based on confidence level
    borderColor = "ff0000";

    // This will be set differently later
    if (confidence > .9){
        borderColor = "00ff00";
    }

    // add the actual area to be highlighted to the map.
    // it's set to not be clickable, to not fill in, and to use the given border color.
    $("#"+imgName+"_map").append($('<area>', {style : {cursor : "default"} ,onclick  : "return false" ,data : {"maphilight" : {fill: false, strokeColor : borderColor}}, class : keyword , shape: "rect", coords: coordinates[0] + "," + coordinates[1] + "," + coordinates[2] + "," + coordinates[3]}));
}


/*
Function that creates something you can scrollover, highlighting all the matches 
*/
function createScrollover(keyword, imgName) {

    // add the div that can be scrolled over. later can be something else instead.
    $("#img-container").append($("<div>", { id : keyword+"_scrollover", text : keyword}));


    $("#"+keyword+"_scrollover").mouseover(function(e) {
        $('#'+imgName).maphilight();
        $("." + keyword).mouseover();
    }).mouseout(function(e){
        $('#'+imgName).maphilight({ neverOn : true});
        $("." + keyword).mouseout();
    }).click(function(e){
        e.preventDefault();
        $("." + keyword).each(function(){
            var data = $(this).data('maphilight');
            data.alwaysOn = !data.alwaysOn;
            $(this).data('maphilight', data);
        });
    });
}


Dropzone.options.seedDrop = {
  addRemoveLinks: true,
  init: function() {
    this.on("success", function(file, response) {

        console.log(file, response);
        img_ids[file.name] = response.results;
        readURL(response.results);


    });
    this.on("removedfile", function(file) {  
        removeBbox(img_ids[file.name]);
    });
  }
};

function readURL(fileNumber) {

    addNewBboxSelector(fileNumber);
    displayImg(fileNumber);
    
}

function displayImg(fileNumber) {
    $("#" + fileNumber).on("load", function(){
	console.log("#"+fileNumber+' on load function');
        // get the int value number of px. This is assuming it is in px
        container_width = parseFloat($("#container").css("width").split("px")[0]);
        image_width = this.width;

        if (image_width > container_width){
            ratio = container_width / image_width;
            $("#" + this.id + "_div").css("width", container_width + "px");
            $("#" + this.id).css("width", "100%");
        }else {
            ratio = 1.0;
        }
        image_ratio_width[fileNumber] = [ratio,image_width];

    });
    $("#"+fileNumber).attr('src', '/blob/'+fileNumber);

}

function addNewBboxSelector(fileNumber) {
    createBbox(fileNumber, "img-container");
    addBboxHandles(fileNumber);

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

function addBboxHandles(fileName) {
    $("#"+fileName).imgAreaSelect({
        handles: true,
        aspectRatio: "1:1",
        parent: "#"+fileName+"_div",
        onSelectEnd: function (img, selection) {
            ratio = image_ratio_width[fileName][0];
                if(typeof user_patches[fileName] === 'undefined'){
                    user_patches[fileName] = [[Math.floor(selection.x1 / ratio), Math.floor(selection.y1 / ratio),Math.floor( selection.width / ratio)]];
                }
                else{
                    user_patches[fileName].push([Math.floor(selection.x1 / ratio), Math.floor(selection.y1 / ratio), Math.floor(selection.width / ratio)]);
                }   
            createSeedPreview(fileName);
        }
    });
}

function createSeedPreview(fileName) {
    count = user_patches[fileName].length-1;
    r = Math.floor((Math.random() * 100) + 1);
    $("#keyword-container").append($('<div>', { id: fileName+"_"+count+"_"+r, class: "bbox"}));   
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
    }
    else {
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

