var colors = ["#d1a3a3", "#d1cea3", "#a5d1a3", "#a3d1cc", "#a3b1d1", "#baa3d1", "#d1a3bf", "#d48585"]; /* section colors */
var color_n = 0;
var curr_section = 1;
var block;

window.addEventListener('load', function () {
  var layout = document.getElementById('layout');
  var layout_els = layout.getElementsByTagName('*');

  var i;
  for(i=0; i < layout_els.length; i++) {
    block = layout_els[i];
    section = block.className.split(' ')[0];

    if (section) {
      if (section != curr_section) {
        curr_section = section;
        color_n++;
        if (color_n == colors.length) {
          color_n = 0;
        }
      }

      color = colors[color_n];
      block.style.borderLeft = "10px solid".concat(color);
    }
  };

  var colls = document.getElementsByClassName("collapsible");

  for(i=0; i < colls.length; i++) {
    colls[i].addEventListener("click", function() {
      this.classList.toggle("active");
      var content = this.nextElementSibling;
      if (content.style.display === "block") {
        // TODO: add small animation with the display styling if possible
        // TODO: along with changing the display, you should add or remove (toggle) the class to style the icons in front of each item in the TOC
        content.style.display = "none";
      }
      else {
        content.style.display = "block";
      }
    });
  }

    // TODO: here you should add all elements in a div or something which is static in the html files!

})

