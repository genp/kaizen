function img_select(){
  $("img").click(function(){
    $(this).toggleClass("overlay");
    console.log(this);
  });


  $("#commit").click(function(){
    event.preventDefault()
    // $(this).prop("disabled", true);

    range = [];

    $('#gallery .patch').each(function(i,img) {
      choosen = $(img).hasClass('overlay') == true;
      pid = img.id.split("-")[1];
      if (choosen) {
        range.push(pid);
      }
    });

    console.log(range);
    console.log(range[0]);
    console.log(range[1]);

    if (range.length > 2){
      alert("You picked too many objects!");
    } else if (range.length < 2){
      alert("You picked too few objects!");
    }  else{
      $("#first_incorrect").val(range[0]);
      $("#last_correct").val(range[1]);

      var data = $("#range_response").serialize();
      var that = this;

      $.ajax({
        type: "POST",
        url: "/classifier/eval_range",
        data: data,
        success: function(response){
          img_first_incorrect = "<img id='patch-"+ range[0]+ "'" + "class='patch' src='/patch/" + range[0] + "'>";
          img_last_correct = "<img id='patch-"+ range[1]+ "'" + "class='patch' src='/patch/" + range[1] + "'>";
          $("#picks").css("height", "250px");
          $("#first_incorrect_div").html("<p>First incorrect</p>" + img_first_incorrect);
          $("#last_correct_div").html("<p>Last Correct</p>" + img_last_correct);
          $("#score_div").html("<h3> Score: " + Math.abs(range[0] - range[1]) + "</h3>")
          $("#space").html("<hr>");

          // update forms
          $("note").val(response["note_id"]);

        }
      });
    }

  });


}
