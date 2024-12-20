<template>
  <div class="legal-query">
    <h1>法律条文查询</h1>
    <div class="search-box">
      <input
        type="text"
        v-model="query"
        placeholder="请输入关键词"
        @keyup.enter="searchLaw"
      />
      <button @click="searchLaw">查询</button>
    </div>
    <div v-if="loading" class="loading">查询中...</div>
    <div v-else-if="results.length > 0" class="results">
      <h2>查询结果：</h2>
      <ul>
        <li v-for="(result, index) in results" :key="index">
          <strong>{{ result.title }}</strong>
          <p>{{ result.content }}</p>
        </li>
      </ul>
    </div>
    <div v-else-if="query" class="no-results">
      未找到相关法律条文。
    </div>
  </div>
</template>

<script>
import axios from "axios";

export default {
  name: "LegalQuery",
  data() {
    return {
      query: "",
      results: [],
      loading: false,
    };
  },
  methods: {
    async searchLaw() {
      if (!this.query) {
        alert("请输入查询关键词！");
        return;
      }
      this.loading = true;
      this.results = [];
      try {
        var data = {"query": this.query, "index_name": "law"}
        // 示例 API 请求，请根据实际修改
        const response = await axios.post(
          `localhost:8000/search`,
          data
        );
        this.results = response.data || [];
      } catch (error) {
        console.error("查询失败：", error);
        alert("查询失败，请稍后重试！");
      } finally {
        this.loading = false;
      }
    },
  },
};
</script>

<style scoped>
.legal-query {
  max-width: 600px;
  margin: 50px auto;
  font-family: Arial, sans-serif;
}
.search-box {
  display: flex;
  gap: 10px;
  margin-bottom: 20px;
}
input {
  flex: 1;
  padding: 10px;
  font-size: 16px;
}
button {
  padding: 10px 20px;
  font-size: 16px;
  background-color: #007bff;
  color: white;
  border: none;
  cursor: pointer;
}
button:hover {
  background-color: #0056b3;
}
.loading {
  font-size: 18px;
  color: gray;
}
.results {
  margin-top: 20px;
}
.results h2 {
  margin-bottom: 10px;
}
.results ul {
  list-style: none;
  padding: 0;
}
.results li {
  margin-bottom: 20px;
}
.no-results {
  font-size: 16px;
  color: red;
}
</style>
