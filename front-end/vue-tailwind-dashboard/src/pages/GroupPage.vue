<template>
  <div class="overflow-x-auto relative sm:rounded-lg px-4 py-4">
    <!-- 搜索框和按钮 -->
    <div class="w-full items-center mb-4">
        <div class="relative">
            <input type="text" v-model="searchQuery" placeholder="请输入关键词..." class="w-full border rounded px-4 py-2 pl-10 placeholder-gray-400">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.5 15.5l2.5 2.5m-5.25-1.5a5.5 5.5 0 111.732-1.732 5.5 5.5 0 01-1.732 1.732z" />
            </svg>
        </div>
    </div>

    <!-- 表格 -->
    <table class="w-full border-collapse mb-4">
      <thead>
        <tr class="bg-gray-200">
          <th class="border px-4 py-2">群组 ID</th>
          <th class="border px-4 py-2">标题</th>
          <th class="border px-4 py-2">用户名</th>
          <th class="border px-4 py-2">群组成员数量</th>
          <th class="border px-4 py-2">更新时间</th>
          <th class="border px-4 py-2">简介</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="(group, index) in displayedItems" :key="index" class="hover:bg-gray-100">
          <td class="border px-4 py-2" v-html="highlightResult(group.channel_id)"></td>
          <td class="border px-4 py-2" v-html="highlightResult(group.title)"></td>
          <td class="border px-4 py-2" v-html="highlightResult(group.username)"></td>
          <td class="border px-4 py-2" v-html="highlightResult(group.participants_count)"></td>
          <td class="border px-4 py-2" v-html="highlightResult(group.update_time)"></td>
          <td class="border px-4 py-2">
            <div class="tooltip" :title="group.entity_info">
              <span v-html="truncateText(highlightResult(group.entity_info))"></span>
            </div>
          </td>
        </tr>
      </tbody>
    </table>

    <!-- 翻页按钮和页码 -->
    <div class="flex justify-center items-center mb-4">
      <button @click="previousPage" :disabled="currentPage === 1" class="bg-gray-700 text-white px-4 py-2 rounded-l hover:bg-gray-500 disabled:bg-gray-300">上一页</button>
      <span class="mx-4">{{ currentPage }} / {{ totalPages }}</span>
      <button @click="nextPage" :disabled="currentPage === totalPages" class="bg-gray-700 text-white px-4 py-2 rounded-r hover:bg-gray-500 disabled:bg-gray-300">下一页</button>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      searchQuery: '', // 搜索关键字
      currentPage: 1, // 当前页数
      itemsPerPage: 6, // 每页显示的条目数
      groups: [], // 从后端获取的群组数据
    };
  },
  computed: {
    // 根据搜索关键字过滤数据
    filteredItems() {
      return this.groups.filter(item =>
        Object.values(item).some(value =>
          value && value.toString().toLowerCase().includes(this.searchQuery.toLowerCase())
        )
      );
    },
    // 计算总页数
    totalPages() {
      return Math.ceil(this.filteredItems.length / this.itemsPerPage);
    },
    // 根据当前页数和每页显示的条目数计算显示的数据
    displayedItems() {
      const startIndex = (this.currentPage - 1) * this.itemsPerPage;
      const endIndex = startIndex + this.itemsPerPage;
      return this.filteredItems.slice(startIndex, endIndex);
    }
  },
  mounted() {
    // 组件加载时从后端获取数据
    this.fetchGroups();
  },
  methods: {
    // 从后端获取群组数据
    fetchGroups() {
      fetch('http://127.0.0.1:5000/fetch_groups')
        .then(response => response.json())
        .then(data => {
          this.groups = data;
          console.log('Groups:', this.groups);
        })
        .catch(error => console.error('Error fetching groups:', error));
    },
    // 上一页
    previousPage() {
      if (this.currentPage > 1) {
        this.currentPage--;
      }
    },
    // 下一页
    nextPage() {
      if (this.currentPage < this.totalPages) {
        this.currentPage++;
      }
    },
    // 截断文本以适应 tooltip
    truncateText(text) {
      const maxLength = 50;
      if (text.length > maxLength) {
        return text.slice(0, maxLength) + '...';
      }
      return text;
    },
    // 在结果中高亮显示搜索关键字
    highlightResult(text) {
      if (!text || !this.searchQuery) return text;
      return text.toString().replace(new RegExp(this.searchQuery, 'gi'), match => `<span class="highlight">${match}</span>`);
    }
  }
};
</script>

<style>
  .tooltip {
    position: relative;
    display: inline-block;
    cursor: pointer;
  }

  .highlight {
    background-color: yellow;
    font-weight: bold;
  }
</style>
