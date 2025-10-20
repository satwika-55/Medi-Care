import React from "react";
import doctorImg from "../assets/doctor.png"; // update path if needed

const Home = () => {
  return (
    <div className="fixed inset-0 min-h-screen w-full bg-[linear-gradient(120deg,_#ffffff_20%,_#6953F0_80%)] flex items-center justify-between px-16 py-10">
      {/* Left Side - Doctor Image */}
      <div className="flex-1 flex justify-start">
        <img
          src={doctorImg}
          alt="Doctor"
          className="w-[950px] h-auto object-contain drop-shadow-2xl mix-blend-multiply bg-transparent"
        />
      </div>

      {/* Right Side - Text Content */}
      <div className="flex-1 flex flex-col items-start justify-center text-left space-y-8 pr-12">
        {/* Tagline */}
        <h2 className="text-[90px] font-extrabold text-[#3a115a] leading-none drop-shadow-md">
          #1
        </h2>

        {/* Main Heading */}
        <h1 className="text-[58px] font-extrabold leading-tight text-[#4B2FE3] drop-shadow-md">
          Doctors brought to your home by{" "}
          <span className="text-[#cbd722]">MediCare</span>
        </h1>

        {/* Description */}
        <p className="text-xl text-white max-w-lg leading-relaxed tracking-wide">
          Upload your problem either through <b>text</b>, <b>image</b>, or even{" "}
          <b>voice</b> — our smart AI instantly connects you with the best
          available doctor for your needs.
        </p>

        {/* Button */}
        <button className="mt-4 px-10 py-4 bg-[#cbd722] text-[#4B2FE3] font-bold rounded-2xl text-lg shadow-lg hover:bg-yellow-300 hover:scale-105 transition-transform duration-300">
          FAQ’S
        </button>
      </div>
    </div>
  );
};

export default Home;
