const express = require('express');
const app = express();
const port = 5000
const cors = require('cors');


app.use(cors()); // <-- this enables CORS for all origins by default



app.post('/submit', (req, res) => {
   console.log(req.body) // your implementation here
   res.send('Saved!')
 });

app.listen(port, () => {
  console.log(`Example app listening at http://localhost:${port}`)
});


