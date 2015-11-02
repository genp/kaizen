$(function() {

  function sticky_relocate() {
    var window_top = $(window).scrollTop();
    var div_top = $('#container-panel').offset().top;
    if (window_top > div_top)
      $('#exemplar-container').addClass('sticky')
    else
      $('#exemplar-container').removeClass('sticky');
  }

  $(window).scroll(sticky_relocate);
  sticky_relocate();

});
