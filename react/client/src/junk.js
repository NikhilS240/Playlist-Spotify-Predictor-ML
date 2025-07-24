// import React, { useState, useEffect } from 'react'
// import { useForm } from "react-hook-form"



//   function MyForm() {
//   return (
//     <form>
//       <label>Please provide the Spotify link you want to use:
//         <input type="text" />
//       </label>
//     </form>
//   )
// }




//   function App() {
//       const [data, setData] = useState(null);

//       useEffect(() => {
//         fetch('http://localhost:5000/data') // Assuming Flask is running on the same origin or you've configured a proxy
//           .then(response => response.json()) //makes js obj
//           .then(data => setData(data)) //put the response into state box
//           .catch(error => console.error('Error fetching data:', error));
//       }, []); //empty array means only run one time

//       return (
//         <div>
//         {data ? (
//           <ul>
//             <li>Name: {data.Name}</li>
//             <li>Age: {data.Age}</li>
//             <li>Programming: {data.programming}</li>
//           </ul>
//         ) : (
//           <p>Loading...</p>
//         )}
//       </div>

      
      
//       )



      
// }




// export default App















// // import { useState } from 'react';



// export default function MyForm() {
//   function handleSubmit(e) {
//     // Prevent the browser from reloading the page
//     e.preventDefault();

//     // Read the form data
//     const form = e.target;
//     const formData = new FormData(form);

//     // You can pass formData as a fetch body directly:

//     // Or you can work with it as a plain object:
//     const formJson = Object.fromEntries(formData.entries());
//     console.log(formJson);
//   }

//   return (
//   <form method="post" onSubmit={handleSubmit}>
//     <label>
//       Text input: 
//       <input name="myInput"  />
//     </label>
//     <br />
//     <button type="submit">Submit</button>
//   </form>
// );
// }










import { useState } from 'react';

export default function Printer() {
  const [count, setCount] = useState('');

  return (
    <div>
      <MyForm setCount={setCount} />
      <p>{count}</p>
    </div>
  );
}

function MyForm({ setCount }) {
  function handleSubmit(e) {
    e.preventDefault();

    const form = e.target;
    const formData = new FormData(form);
    const formJson = Object.fromEntries(formData.entries());
    console.log(formJson);

    setCount(formJson.myInput);
  }

  return (
    <form method="post" onSubmit={handleSubmit}>
      <label>
        Text input:
        <input name="myInput" />
      </label>
      <br />
      <button type="submit">Submit</button>
    </form>
  );
}
