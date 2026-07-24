graph [
  directed 1
  node [
    id 0
    label "Latitude"
  ]
  node [
    id 1
    label "Longitude"
  ]
  node [
    id 2
    label "Temperature"
  ]
  node [
    id 3
    label "Radiation"
  ]
  node [
    id 4
    label "Soil Properties"
  ]
  node [
    id 5
    label "Climate Water Balance"
  ]
  node [
    id 6
    label "Soil Moisture"
  ]
  node [
    id 7
    label "NDVI"
  ]
  node [
    id 8
    label "FPAR"
  ]
  node [
    id 9
    label "Agricultural Intensity"
  ]
  node [
    id 10
    label "Year"
  ]
  node [
    id 11
    label "Yield"
  ]
  edge [
    source 0
    target 2
  ]
  edge [
    source 0
    target 3
  ]
  edge [
    source 0
    target 11
  ]
  edge [
    source 1
    target 2
  ]
  edge [
    source 1
    target 3
  ]
  edge [
    source 1
    target 11
  ]
  edge [
    source 2
    target 5
  ]
  edge [
    source 2
    target 6
  ]
  edge [
    source 2
    target 7
  ]
  edge [
    source 2
    target 11
  ]
  edge [
    source 3
    target 7
  ]
  edge [
    source 3
    target 11
  ]
  edge [
    source 4
    target 6
  ]
  edge [
    source 4
    target 11
  ]
  edge [
    source 5
    target 6
  ]
  edge [
    source 5
    target 11
  ]
  edge [
    source 6
    target 7
  ]
  edge [
    source 6
    target 8
  ]
  edge [
    source 6
    target 11
  ]
  edge [
    source 7
    target 11
  ]
  edge [
    source 8
    target 11
  ]
  edge [
    source 9
    target 11
  ]
  edge [
    source 10
    target 6
  ]
  edge [
    source 10
    target 11
  ]
]
