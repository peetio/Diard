var colors = ["#d1a3a3", "#d1cea3", "#a5d1a3", "#a3d1cc", "#a3b1d1", "#baa3d1", "#d1a3bf", "#d48585"]; /* section colors */
var color_n = 0;
var curr_section = 0;
var block;

function addSectionToTOC(text, color, table) {
  let container = document.createElement("div");
  let button = document.createElement("button");
  button.type = "button";
  button.classList.add("collapsible");

  let icon_p = document.createElement("p");
  let icon = document.createElement("span");
  icon.classList.add("material-icons");
  icon.innerHTML = "expand_more";
  icon.style.color = color;
  icon_p.appendChild(icon);
  button.appendChild(icon_p);

  let title = document.createElement("p");
  title.innerHTML = text;
  button.appendChild(title);

  let hr = document.createElement("hr");
  button.appendChild(hr);
  container.appendChild(button);

  let sub_container = document.createElement("div");
  sub_container.classList.add("content");
  container.appendChild(sub_container);

  table.appendChild(container);
}

function addSubSectionToTOC(text, sub_section) {
  let sub_title = document.createElement("p");
  sub_title.innerHTML = text;
  sub_section.appendChild(sub_title);

  let hr = document.createElement("hr");
  sub_section.appendChild(hr);
  console.log(sub_section);
}



// TODO: create another function to add sub titles in the sub_container (content div), don't forget hr tag after each

window.addEventListener('load', function () {
  var layout = document.getElementById('layout');
  var layout_els = layout.getElementsByTagName('*');
  var table = document.body.getElementsByClassName('table')[0];

  var i;
  for(i=0; i < layout_els.length; i++) {
    block = layout_els[i];
    // TODO: add id to each for TOC item
    // TODO: change the icons depending on the situation and clicks
    section = block.className.split(' ')[0];
    tag = block.tagName;
    text = block.innerHTML;
    toc_items = table.getElementsByClassName('content');
    sub_section = toc_items[toc_items.length-1];

    if (section) {
      if (section != curr_section) {
        curr_section = section;
        color_n++;
        if (color_n == colors.length) {
          color_n = 0;
        }
        if (tag == 'H2'){
          addSectionToTOC(text, colors[color_n], table);
        }
      }
      else {
        if (tag == 'H2'){
          if (typeof sub_section !== 'undefined') {
            addSubSectionToTOC(text, sub_section);
          }
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
        content.style.display = "none";
      }
      else {
          content.style.display = "block";
        }
    });
  }
})

