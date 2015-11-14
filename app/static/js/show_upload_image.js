var user_patches = {};
var img_ids = {}

Dropzone.options.seedDrop = {
  addRemoveLinks: true,
  init: function() {
    this.on("success", function(file, response) {
        console.log(file, response);

        readURL(response.results);
    });
    this.on("removedfile", function(file) {  
        removeBbox(file);
    });
  }
};

Dropzone.options.datasetDrop = {
    maxFiles: 1,
    addRemoveLinks: true, 
  init: function() {
    this.on("success", function(file, response) {

        //console.log(file, response);
        console.log("In datasetDrop");
        console.log(response);

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
    // make the actual image element
    $("#"+imgName+"_div").html($('<img>', { id: imgName, usemap: "#"+imgName+"_map" }));
    // create the map for the image, used later to highlight patches
    $("#"+imgName+"_div").append($('<map>', {id: imgName+"_map", name: imgName + "_map"}));
    // load in the image to the img element
    $("#"+imgName).attr('src', '/blobs/'+file);


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

function readURL(file) {

     imgName = "bbox-"+Math.floor(Math.random()*1000).toString();
     createBbox(imgName);
     createSeedPreview(imgName);
     img_ids[file] = imgName;
     addBboxHandles(file);


     $("#"+imgName).on("load", function(){
        console.log(this.width);
     });

     $("#"+imgName).attr('src', '/blobs/'+file);
    
}

function createBbox(imgName) {

    $("#img-container").append($('<div>', { id: imgName+"_div", class: "bbox"}));
    $("#"+imgName+"_div").html($('<img>', { id: imgName }));
}

function removeBbox(fileName) {
    $("#"+img_ids[fileName]).imgAreaSelect({remove:true});
    $("#"+img_ids[fileName]).hide();
    if( user_patches.length <= 0 ) {
        if($.inArray(fileName, user_patches.keys())) {
            delete user_patches[fileName];
        }
    }

    removeSeedPreview(fileName);
}

function addBboxHandles(fileName) {
    imgName = img_ids[fileName];
    $("#"+imgName).imgAreaSelect({
        handles: true,
        aspectRatio: "1:1",
        parent: "#"+imgName+"_div",
        onSelectEnd: function (img, selection) {
            user_patches[fileName] = [selection.y1, selection.x1, selection.width];
            updateSeedPreview(fileName);
        }
    });
}

function createSeedPreview(imgName) {
    $("#keyword-container").prepend($('<div>', { id: imgName+"_seed", class: "bbox"}));    

}

function updateSeedPreview(fileName) {
    imgName = img_ids[fileName];
    img_width = $("#"+imgName).width();
    x = user_patches[fileName][0];
    y = user_patches[fileName][1];
    size = user_patches[fileName][2];
    
    

    $("#"+imgName+"_seed").css({ "background-image": "url(" + $("#"+imgName).attr("src") + ")",  "width" : "100px", "height" : "100px",  "background-size": 100.0*img_width/size+"px", "background-position":"-"+100.0*y/size+"px -"+100.0*x/size+"px",  "border": "2px solid #222"});

    console.dir(JSON.stringify(user_patches));
}

function removeSeedPreview(fileName) {

    $("#"+img_ids[fileName]+"_seed").hide();

}



