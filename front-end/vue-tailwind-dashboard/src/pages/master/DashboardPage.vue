<template>
  <div class="w-screen h-screen flex">
    <!-- Side Bar -->
    <div class="w-[200px] h-full bg-gray-200">
      <div class="h-[60px] bg-gray-800 text-white flex justify-center items-center">
        <!-- Title -->
        <h3 class="text-center font-bold text-xl">Admin Dashboard</h3>
      </div>
      <div class="h-[calc(100%-60px)] bg-gray-700 py-[20px]">
        <!-- Navigation -->
        <div class="px-[20px] flex flex-col space-y-[20px]">
          <router-link to="/group" class="inline-flex relative items-center rounded-lg py-[10px] px-[10px] w-full text-sm text-white bg-gray-700 hover:bg-gray-500">
            <svg class="w-[20px] h-[20px] mr-[10px]" fill="currentColor" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path>
            </svg>
            群组
          </router-link>

          <router-link to="/user" class="inline-flex relative items-center rounded-lg py-[10px] px-[10px] w-full text-sm text-white bg-gray-700 hover:bg-gray-500">
            <svg class="w-[20px] h-[20px] mr-[10px]" fill="currentColor" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path>
            </svg>
            用户
          </router-link>

          <router-link to="/account" class="inline-flex relative items-center rounded-lg py-[10px] px-[10px] w-full text-sm text-white bg-gray-700 hover:bg-gray-500">
            <svg class="w-[20px] h-[20px] mr-[10px]" fill="currentColor" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path>
            </svg>
            账号关联查询
          </router-link>

          <router-link to="/session" class="inline-flex relative items-center rounded-lg py-[10px] px-[10px] w-full text-sm text-white bg-gray-700 hover:bg-gray-500">
            <svg class="w-[20px] h-[20px] mr-[10px]" fill="currentColor" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path>
            </svg>
            会话关联查询
          </router-link>

          <router-link to="/friend" class="inline-flex relative items-center rounded-lg py-[10px] px-[10px] w-full text-sm text-white bg-gray-700 hover:bg-gray-500">
            <svg class="w-[20px] h-[20px] mr-[10px]" fill="currentColor" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path>
            </svg>
            好友关系查询
          </router-link>

          <router-link to="/transaction" class="inline-flex relative items-center rounded-lg py-[10px] px-[10px] w-full text-sm text-white bg-gray-700 hover:bg-gray-500">
            <svg class="w-[20px] h-[20px] mr-[10px]" fill="currentColor" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path>
            </svg>
            交易关系查询
          </router-link>
        </div>
      </div>
    </div>
    <!-- Main Content -->
    <div class="flex-1 h-full bg-gray-100">
      <div class="h-[60px] bg-gray-800 text-white">
      </div>

      <div class="h-[calc(100%-60px)] ">
        <router-view></router-view>
      </div>
    </div>
  </div>
</template>

<script>
export default {

}
</script>

<style>

</style>