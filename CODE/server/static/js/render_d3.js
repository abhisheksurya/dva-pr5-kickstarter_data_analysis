//  render slopegraph chart

(function() {

 // create chart
 var slopegraph = d3.eesur.slopegraph()
                 // .margin({top: 20, bottom: 20, left: 100, right:100})
                    .strokeColour('#130C0E')
                    .keyName('Feature')
                    .keyValueStart('Campaign')
                    .keyValueEnd('Feedback')
                    .h(500)
                    // .format(d3.format('04d'))
                    .on('_hover', function (d, i) {
                            highlightLine(d, i);
                    });


 d3.json('../../samples/data.json', function(error, data) {
    if (error) throw error;
    
    // render chart
    d3.select('#slopegraph')
      .datum(data)
      .call(slopegraph);
    
    // alternative navigation
    navAlt(data);
});

 // reset button listener
 d3.select('#reset')
   .on('click', function () {
        d3.selectAll('.elm').transition().style('opacity', 1);
        d3.selectAll('.s-line').style('stroke', '#130C0E');
   });

 // navigation
 function navAlt(data) {
     d3.select('#nav-alt').append('ul')
       .selectAll('li')
       .data(data)
       .enter().append('li')
       .on('click', function (d, i) {
            highlightLine(d, i);
       })
       .text(function (d) { return d['Feature']; });
 }

 // highlight line and fade other lines
 function highlightLine(d, i) {
     d3.selectAll('.elm').transition().style('opacity', 0.2);
     d3.selectAll('.sel-' + i).transition().style('opacity', 1);
     d3.selectAll('.s-line').style('stroke', '#FDBB30');
     d3.selectAll('.s-line.sel-' + i).style('stroke', '#130C0E');
 }

 // just for blocks viewer size
 d3.select(self.frameElement).style('height', '800px');

}());
